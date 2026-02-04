//! egg-bridge: A C/C++ FFI bridge for the egg e-graph library
//!
//! This crate provides a C-compatible interface for using egg's equality
//! saturation capabilities from C++ code (specifically for MLIR dialects).

use egg::{rewrite as rw, *};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

// Define a simple expression language that can represent arithmetic operations
// This can be extended to match Neura dialect operations
define_language! {
    /// A simple language for representing expressions
    pub enum SimpleExprLang {
        // Constants
        Num(i64),
        Float(ordered_float::NotNan<f64>),
        Symbol(Symbol),

        // Binary arithmetic operations
        "add" = Add([Id; 2]),
        "sub" = Sub([Id; 2]),
        "mul" = Mul([Id; 2]),
        "div" = Div([Id; 2]),

        // Unary operations
        "neg" = Neg(Id),
        
        // Comparison operations
        "==" = Eq([Id; 2]),
        "!=" = Ne([Id; 2]),
        "<" = Lt([Id; 2]),
        "<=" = Le([Id; 2]),
        ">" = Gt([Id; 2]),
        ">=" = Ge([Id; 2]),
        "icmp" = Icmp([Id; 3]),  // Integer comparison with mode

        // Logical operations
        "and" = And([Id; 2]),
        "or" = Or([Id; 2]),
        "not" = Not(Id),

        // Control flow / selection
        "select" = Select([Id; 3]),
        "br" = Br([Id; 3]),        // Branch: (br cond true_target false_target)
        
        // Fused control flow operations
        "br_cmp" = BrCmp([Id; 5]),              // Fused branch with compare: (br_cmp mode a b true false)
        "mul_add_icmp" = MulAddIcmp([Id; 5]),   // Fused mul->add->icmp: (mul_add_icmp mode a b c d) = icmp(mode, a*b+c, d)

        // Memory operations (for Neura dialect)
        "load" = Load(Id),
        "store" = Store([Id; 2]),
        "gep" = Gep(Box<[Id]>),  // Variable arity to support 2 or 3 operands

        // Neura dialect specific operations
        // Use Box<[Id]> for variable arity to support both nullary and unary versions
        "grant_once" = GrantOnce(Box<[Id]>),
        "reserve" = Reserve(Box<[Id]>),
        "phi_start" = PhiStart([Id; 2]),
        "phi_end" = PhiEnd([Id; 2]),
        "grant_pred" = GrantPred([Id; 2]),
        "ctrl_mov" = CtrlMov([Id; 2]),
        "data_mov" = DataMov(Id),
        "return_value" = ReturnValue(Id),
        "return_void" = ReturnVoid(Id),  // Return void with predicate
        "yield" = Yield,
        
        // Floating point operations
        "fadd" = FAdd([Id; 2]),
        "fsub" = FSub([Id; 2]),
        "fmul" = FMul([Id; 2]),
        "fdiv" = FDiv([Id; 2]),

        // Vector operations
        "vadd" = VAdd([Id; 2]),
        "vmul" = VMul([Id; 2]),

        // Type conversion operations
        "sext" = Sext(Id),         // Sign extend
        "zext" = Zext(Id),         // Zero extend
        "trunc" = Trunc(Id),       // Truncate
        "bitcast" = Bitcast(Id),   // Bitcast
        
        // Memory allocation
        "alloca" = Alloca(Id),     // Stack allocation
        
        // Constant (for symbolic constants)
        "const" = Const(Id),       // Symbolic constant
        
        // Function return
        "return" = Return(Id),     // Return from function

        // Nullary operations (no arguments)
        "grant_once_nullary" = GrantOnceNullary,  // grant_once without arguments
        
        // Generic function call with variable arity
        "call" = Call(Box<[Id]>),
        
        // Fused operations (dynamically matched)
        "fused" = Fused(Box<[Id]>),
    }
}

/// Configuration for the equality saturation runner
#[repr(C)]
pub struct EggConfig {
    /// Maximum number of iterations
    pub iter_limit: u32,
    /// Maximum number of nodes in the e-graph
    pub node_limit: u32,
    /// Time limit in seconds (0 = no limit)
    pub time_limit_secs: u32,
}

impl Default for EggConfig {
    fn default() -> Self {
        EggConfig {
            iter_limit: 30,
            node_limit: 10000,
            time_limit_secs: 60,
        }
    }
}

/// Create a default configuration
#[no_mangle]
pub extern "C" fn egg_config_default() -> EggConfig {
    EggConfig::default()
}

/// Parse rewrite rules from a string format
/// Format: "name: lhs => rhs" or "name: lhs <=> rhs" for bidirectional
fn parse_rules(rules_str: &str) -> Result<Vec<Rewrite<SimpleExprLang, ()>>, String> {
    let mut rules = Vec::new();
    
    for line in rules_str.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        
        // Parse "name: lhs => rhs" or "name: lhs <=> rhs"
        let parts: Vec<&str> = line.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid rule format: {}", line));
        }
        
        let name = parts[0].trim();
        let rule_part = parts[1].trim();
        
        if rule_part.contains("<=>") {
            // Bidirectional rule
            let sides: Vec<&str> = rule_part.split("<=>").collect();
            if sides.len() != 2 {
                return Err(format!("Invalid bidirectional rule: {}", line));
            }
            let lhs = sides[0].trim();
            let rhs = sides[1].trim();
            
            let lhs_pattern: Pattern<SimpleExprLang> = lhs.parse()
                .map_err(|e| format!("Failed to parse LHS '{}': {:?}", lhs, e))?;
            let rhs_pattern: Pattern<SimpleExprLang> = rhs.parse()
                .map_err(|e| format!("Failed to parse RHS '{}': {:?}", rhs, e))?;
            
            rules.push(Rewrite::new(
                format!("{}-fwd", name),
                lhs_pattern.clone(),
                rhs_pattern.clone(),
            ).map_err(|e| format!("Failed to create rule {}: {}", name, e))?);
            
            rules.push(Rewrite::new(
                format!("{}-bwd", name),
                rhs_pattern,
                lhs_pattern,
            ).map_err(|e| format!("Failed to create rule {}: {}", name, e))?);
        } else if rule_part.contains("=>") {
            // Unidirectional rule
            let sides: Vec<&str> = rule_part.split("=>").collect();
            if sides.len() != 2 {
                return Err(format!("Invalid rule: {}", line));
            }
            let lhs = sides[0].trim();
            let rhs = sides[1].trim();
            
            let lhs_pattern: Pattern<SimpleExprLang> = lhs.parse()
                .map_err(|e| format!("Failed to parse LHS '{}': {:?}", lhs, e))?;
            let rhs_pattern: Pattern<SimpleExprLang> = rhs.parse()
                .map_err(|e| format!("Failed to parse RHS '{}': {:?}", rhs, e))?;
            
            rules.push(Rewrite::new(
                name,
                lhs_pattern,
                rhs_pattern,
            ).map_err(|e| format!("Failed to create rule {}: {}", name, e))?);
        } else {
            return Err(format!("Rule must contain '=>' or '<=>': {}", line));
        }
    }
    
    Ok(rules)
}

/// Free a string allocated by the Rust library
#[no_mangle]
pub extern "C" fn egg_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

/// Get the version of the egg-bridge library
#[no_mangle]
pub extern "C" fn egg_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Result structure for cycle-aware extraction
#[repr(C)]
pub struct EggCycleResult {
    /// The optimized expression as a string (owned, must be freed)
    pub result_expr: *mut c_char,
    /// Error message if any (owned, must be freed), null if success
    pub error_msg: *mut c_char,
    /// Number of iterations performed
    pub iterations: u32,
    /// Final e-graph size (number of e-classes)
    pub egraph_size: u32,
    /// Whether saturation was reached
    pub saturated: bool,
    /// Number of nodes on cycles in the extracted expression
    pub cycle_node_count: u32,
    /// Total area of nodes NOT on cycles (for phase 2 optimization)
    pub off_cycle_area: u32,
    /// Total AST size of the extracted expression
    pub ast_size: u32,
}

/// Check if a node is a backward edge (ctrl_mov creates cycles)
fn is_back_edge_node(node: &SimpleExprLang) -> bool {
    matches!(node, SimpleExprLang::CtrlMov(_))
}

/// Run equality saturation with cycle-aware extraction
///
/// This extraction method uses two-phase optimization:
/// - Phase 1: Minimize the number of nodes on cycles (critical path)
/// - Phase 2: Minimize total area for nodes NOT on cycles
///
/// Cost semantics:
/// - ctrl_mov operations are treated as backward edges (create cycles)
/// - Only nodes ON cycles contribute to cycle_node_count
/// - Only nodes NOT on cycles contribute to off_cycle_area
/// - Cost comparison: first by cycle_node_count, then by off_cycle_area, then by ast_size
///
/// # Arguments
/// * `expr_str` - The initial expression as a string (S-expression format)
/// * `rules_str` - The rewrite rules as a string (one rule per line)
/// * `config` - Configuration for the runner
///
/// # Returns
/// An EggCycleResult containing the optimized expression with cycle information
#[no_mangle]
pub extern "C" fn egg_run_saturation_cycle_aware(
    expr_str: *const c_char,
    rules_str: *const c_char,
    config: EggConfig,
) -> EggCycleResult {
    // Helper to create error result
    let make_error = |msg: &str| -> EggCycleResult {
        EggCycleResult {
            result_expr: ptr::null_mut(),
            error_msg: CString::new(msg).unwrap().into_raw(),
            iterations: 0,
            egraph_size: 0,
            saturated: false,
            cycle_node_count: 0,
            off_cycle_area: 0,
            ast_size: 0,
        }
    };
    
    // Parse input strings
    let expr_str = unsafe {
        if expr_str.is_null() {
            return make_error("Expression string is null");
        }
        match CStr::from_ptr(expr_str).to_str() {
            Ok(s) => s,
            Err(_) => return make_error("Invalid UTF-8 in expression string"),
        }
    };
    
    let rules_str = unsafe {
        if rules_str.is_null() {
            return make_error("Rules string is null");
        }
        match CStr::from_ptr(rules_str).to_str() {
            Ok(s) => s,
            Err(_) => return make_error("Invalid UTF-8 in rules string"),
        }
    };
    
    // Parse the expression
    let start_expr: RecExpr<SimpleExprLang> = match expr_str.parse() {
        Ok(e) => e,
        Err(e) => return make_error(&format!("Failed to parse expression: {:?}", e)),
    };
    
    // Parse the rules
    let rules = match parse_rules(rules_str) {
        Ok(r) => r,
        Err(e) => return make_error(&format!("Failed to parse rules: {}", e)),
    };
    
    // Create and configure the runner
    let mut runner = Runner::default()
        .with_expr(&start_expr)
        .with_iter_limit(config.iter_limit as usize)
        .with_node_limit(config.node_limit as usize);
    
    if config.time_limit_secs > 0 {
        runner = runner.with_time_limit(std::time::Duration::from_secs(config.time_limit_secs as u64));
    }
    
    // Run equality saturation
    let runner = runner.run(&rules);
    
    // Use cycle-aware extraction (MinCycleExtractor)
    // - is_back_edge_node: identifies ctrl_mov as backward edges for cycle detection
    // - Only nodes ON cycles contribute to cycle_node_count
    // - Only nodes NOT on cycles contribute to off_cycle_area (with default area=1)
    let extractor = MinCycleExtractor::new(&runner.egraph, is_back_edge_node);
    let (cost, best_expr) = extractor.find_best(runner.roots[0]);
    
    // Check if saturated
    let saturated = matches!(runner.stop_reason, Some(StopReason::Saturated));
    
    // Return result with cycle information
    EggCycleResult {
        result_expr: CString::new(best_expr.to_string()).unwrap().into_raw(),
        error_msg: ptr::null_mut(),
        iterations: runner.iterations.len() as u32,
        egraph_size: runner.egraph.number_of_classes() as u32,
        saturated,
        cycle_node_count: cost.cycle_node_count as u32,
        off_cycle_area: cost.off_cycle_area as u32,
        ast_size: cost.ast_size as u32,
    }
}

/// Type for the area callback function
/// 
/// The callback receives the operation name as a C string and returns the area cost.
/// This allows C++ code to provide custom area costs for different operations.
pub type AreaCallback = extern "C" fn(op_name: *const c_char) -> u32;

/// Run equality saturation with cycle-aware extraction and custom area function
///
/// This extraction method uses two-phase optimization:
/// - Phase 1: Minimize the number of nodes on cycles (critical path)
/// - Phase 2: Minimize total area for nodes NOT on cycles
///
/// # Arguments
/// * `expr_str` - The initial expression as a string (S-expression format)
/// * `rules_str` - The rewrite rules as a string (one rule per line)
/// * `config` - Configuration for the runner
/// * `area_callback` - Callback function that returns the area cost for a given operation name
///
/// # Returns
/// An EggCycleResult containing the optimized expression with cycle information
#[no_mangle]
pub extern "C" fn egg_run_saturation_cycle_aware_with_area(
    expr_str: *const c_char,
    rules_str: *const c_char,
    config: EggConfig,
    area_callback: AreaCallback,
) -> EggCycleResult {
    // Helper to create error result
    let make_error = |msg: &str| -> EggCycleResult {
        EggCycleResult {
            result_expr: ptr::null_mut(),
            error_msg: CString::new(msg).unwrap().into_raw(),
            iterations: 0,
            egraph_size: 0,
            saturated: false,
            cycle_node_count: 0,
            off_cycle_area: 0,
            ast_size: 0,
        }
    };
    
    // Parse input strings
    let expr_str = unsafe {
        if expr_str.is_null() {
            return make_error("Expression string is null");
        }
        match CStr::from_ptr(expr_str).to_str() {
            Ok(s) => s,
            Err(_) => return make_error("Invalid UTF-8 in expression string"),
        }
    };
    
    let rules_str = unsafe {
        if rules_str.is_null() {
            return make_error("Rules string is null");
        }
        match CStr::from_ptr(rules_str).to_str() {
            Ok(s) => s,
            Err(_) => return make_error("Invalid UTF-8 in rules string"),
        }
    };
    
    // Parse the expression
    let start_expr: RecExpr<SimpleExprLang> = match expr_str.parse() {
        Ok(e) => e,
        Err(e) => return make_error(&format!("Failed to parse expression: {:?}", e)),
    };
    
    // Parse the rules
    let rules = match parse_rules(rules_str) {
        Ok(r) => r,
        Err(e) => return make_error(&format!("Failed to parse rules: {}", e)),
    };
    
    // Create and configure the runner
    let mut runner = Runner::default()
        .with_expr(&start_expr)
        .with_iter_limit(config.iter_limit as usize)
        .with_node_limit(config.node_limit as usize);
    
    if config.time_limit_secs > 0 {
        runner = runner.with_time_limit(std::time::Duration::from_secs(config.time_limit_secs as u64));
    }
    
    // Run equality saturation
    let runner = runner.run(&rules);
    
    // Define area function that calls the C callback
    let area_fn = move |node: &SimpleExprLang| -> usize {
        // Get the operation name for this node
        let op_name = get_node_op_name(node);
        let c_str = CString::new(op_name).unwrap_or_else(|_| CString::new("unknown").unwrap());
        area_callback(c_str.as_ptr()) as usize
    };
    
    // Use cycle-aware extraction with custom area function
    let extractor = MinCycleExtractor::new_with_area(&runner.egraph, is_back_edge_node, area_fn);
    let (cost, best_expr) = extractor.find_best(runner.roots[0]);
    
    // Check if saturated
    let saturated = matches!(runner.stop_reason, Some(StopReason::Saturated));
    
    // Return result with cycle information
    EggCycleResult {
        result_expr: CString::new(best_expr.to_string()).unwrap().into_raw(),
        error_msg: ptr::null_mut(),
        iterations: runner.iterations.len() as u32,
        egraph_size: runner.egraph.number_of_classes() as u32,
        saturated,
        cycle_node_count: cost.cycle_node_count as u32,
        off_cycle_area: cost.off_cycle_area as u32,
        ast_size: cost.ast_size as u32,
    }
}

/// Helper function to get the operation name for a node
fn get_node_op_name(node: &SimpleExprLang) -> &'static str {
    match node {
        SimpleExprLang::Num(_) => "num",
        SimpleExprLang::Float(_) => "float",
        SimpleExprLang::Symbol(_) => "symbol",
        SimpleExprLang::Add(_) => "add",
        SimpleExprLang::Sub(_) => "sub",
        SimpleExprLang::Mul(_) => "mul",
        SimpleExprLang::Div(_) => "div",
        SimpleExprLang::Neg(_) => "neg",
        SimpleExprLang::Eq(_) => "eq",
        SimpleExprLang::Ne(_) => "ne",
        SimpleExprLang::Lt(_) => "lt",
        SimpleExprLang::Le(_) => "le",
        SimpleExprLang::Gt(_) => "gt",
        SimpleExprLang::Ge(_) => "ge",
        SimpleExprLang::Icmp(_) => "icmp",
        SimpleExprLang::And(_) => "and",
        SimpleExprLang::Or(_) => "or",
        SimpleExprLang::Not(_) => "not",
        SimpleExprLang::Select(_) => "select",
        SimpleExprLang::Br(_) => "br",
        SimpleExprLang::BrCmp(_) => "br_cmp",
        SimpleExprLang::MulAddIcmp(_) => "mul_add_icmp",
        SimpleExprLang::Load(_) => "load",
        SimpleExprLang::Store(_) => "store",
        SimpleExprLang::Gep(_) => "gep",
        SimpleExprLang::GrantOnce(_) => "grant_once",
        SimpleExprLang::Reserve(_) => "reserve",
        SimpleExprLang::PhiStart(_) => "phi_start",
        SimpleExprLang::PhiEnd(_) => "phi_end",
        SimpleExprLang::GrantPred(_) => "grant_pred",
        SimpleExprLang::CtrlMov(_) => "ctrl_mov",
        SimpleExprLang::DataMov(_) => "data_mov",
        SimpleExprLang::ReturnValue(_) => "return_value",
        SimpleExprLang::ReturnVoid(_) => "return_void",
        SimpleExprLang::Yield => "yield",
        SimpleExprLang::FAdd(_) => "fadd",
        SimpleExprLang::FSub(_) => "fsub",
        SimpleExprLang::FMul(_) => "fmul",
        SimpleExprLang::FDiv(_) => "fdiv",
        SimpleExprLang::VAdd(_) => "vadd",
        SimpleExprLang::VMul(_) => "vmul",
        SimpleExprLang::Sext(_) => "sext",
        SimpleExprLang::Zext(_) => "zext",
        SimpleExprLang::Trunc(_) => "trunc",
        SimpleExprLang::Bitcast(_) => "bitcast",
        SimpleExprLang::Alloca(_) => "alloca",
        SimpleExprLang::Const(_) => "const",
        SimpleExprLang::Return(_) => "return",
        SimpleExprLang::GrantOnceNullary => "grant_once_nullary",
        SimpleExprLang::Call(_) => "call",
        SimpleExprLang::Fused(_) => "fused",
    }
}

/// Free the memory allocated for an EggCycleResult
#[no_mangle]
pub extern "C" fn egg_cycle_result_free(result: EggCycleResult) {
    unsafe {
        if !result.result_expr.is_null() {
            drop(CString::from_raw(result.result_expr));
        }
        if !result.error_msg.is_null() {
            drop(CString::from_raw(result.error_msg));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_complex_two_cycles_with_fusion() {
        // 构造一个包含两个环的复杂图，第一个环比第二个长：
        //
        // 环1 (长): x -> mul -> add -> icmp -> br -> (ctrl_mov 回到 x)
        //          共5个节点在环上: x, mul, add, icmp, br
        //
        // 环2 (短): y -> add1 -> add2 -> icmp -> br -> (ctrl_mov 回到 y)
        //          共5个节点在环上: y, add1, add2, icmp, br
        //
        // 改写规则：
        // 1. fuse_icmp_br: (br (icmp ?mode ?a ?b) ?t ?f) => (br_cmp ?mode ?a ?b ?t ?f)
        //    - 将 icmp 和 br 合并，减少1个环上节点
        // 2. fuse_mul_add_icmp: (icmp ?mode (+ (* ?a ?b) ?c) ?d) => (mul_add_icmp ?mode ?a ?b ?c ?d)
        //    - 将 mul->add->icmp 合并为一个操作，减少2个环上节点
        //
        // 预期结果：
        // - 环1应该使用 mul_add_icmp 融合（因为它能减少更多节点）
        // - 环2只能使用 br_cmp 融合（因为它没有 mul，不匹配 mul_add_icmp 模式）
        
        let mut egraph = EGraph::<SimpleExprLang, ()>::default();
        
        // === 环1的构建: x -> mul -> add -> icmp -> br ===
        let x: RecExpr<SimpleExprLang> = "x".parse().unwrap();
        let x_id = egraph.add_expr(&x);
        
        let const2: RecExpr<SimpleExprLang> = "2".parse().unwrap();
        let const2_id = egraph.add_expr(&const2);
        
        let const3: RecExpr<SimpleExprLang> = "3".parse().unwrap();
        let const3_id = egraph.add_expr(&const3);
        
        let const100: RecExpr<SimpleExprLang> = "100".parse().unwrap();
        let const100_id = egraph.add_expr(&const100);
        
        let true_target1: RecExpr<SimpleExprLang> = "loop1_body".parse().unwrap();
        let true_target1_id = egraph.add_expr(&true_target1);
        
        let false_target1: RecExpr<SimpleExprLang> = "loop1_exit".parse().unwrap();
        let false_target1_id = egraph.add_expr(&false_target1);
        
        // mul1 = x * 2
        let mul1 = egraph.add(SimpleExprLang::Mul([x_id, const2_id]));
        // add1 = mul1 + 3 = x * 2 + 3
        let add1 = egraph.add(SimpleExprLang::Add([mul1, const3_id]));
        // lt_mode 用于 icmp
        let lt_mode: RecExpr<SimpleExprLang> = "lt".parse().unwrap();
        let lt_mode_id = egraph.add_expr(&lt_mode);
        // icmp1 = icmp lt (x*2+3) 100
        let icmp1 = egraph.add(SimpleExprLang::Icmp([lt_mode_id, add1, const100_id]));
        // br1 = br icmp1 true_target1 false_target1
        let br1 = egraph.add(SimpleExprLang::Br([icmp1, true_target1_id, false_target1_id]));
        // ctrl_mov 创建反向边：从 br1 回到 x，形成环
        let ctrl1 = egraph.add(SimpleExprLang::CtrlMov([br1, x_id]));
        
        // === 环2的构建: y -> add1 -> add2 -> icmp -> br ===
        let y: RecExpr<SimpleExprLang> = "y".parse().unwrap();
        let y_id = egraph.add_expr(&y);
        
        let const1: RecExpr<SimpleExprLang> = "1".parse().unwrap();
        let const1_id = egraph.add_expr(&const1);
        
        let const5: RecExpr<SimpleExprLang> = "5".parse().unwrap();
        let const5_id = egraph.add_expr(&const5);
        
        let const50: RecExpr<SimpleExprLang> = "50".parse().unwrap();
        let const50_id = egraph.add_expr(&const50);
        
        let true_target2: RecExpr<SimpleExprLang> = "loop2_body".parse().unwrap();
        let true_target2_id = egraph.add_expr(&true_target2);
        
        let false_target2: RecExpr<SimpleExprLang> = "loop2_exit".parse().unwrap();
        let false_target2_id = egraph.add_expr(&false_target2);
        
        // add2_1 = y + 1
        let add2_1 = egraph.add(SimpleExprLang::Add([y_id, const1_id]));
        // add2_2 = add2_1 + 5 = y + 1 + 5
        let add2_2 = egraph.add(SimpleExprLang::Add([add2_1, const5_id]));
        // icmp2 = icmp lt (y+1+5) 50
        let icmp2 = egraph.add(SimpleExprLang::Icmp([lt_mode_id, add2_2, const50_id]));
        // br2 = br icmp2 true_target2 false_target2
        let br2 = egraph.add(SimpleExprLang::Br([icmp2, true_target2_id, false_target2_id]));
        // ctrl_mov 创建反向边：从 br2 回到 y，形成环
        let ctrl2 = egraph.add(SimpleExprLang::CtrlMov([br2, y_id]));
        
        // 创建一个根节点，包含两个环的结果
        let root = egraph.add(SimpleExprLang::Add([ctrl1, ctrl2]));
        
        egraph.rebuild();
        
        // 定义融合规则
        let rules_str = r#"
            # 规则1: 将 icmp 和 br 融合为 br_cmp
            # (br (icmp mode a b) t f) => (br_cmp mode a b t f)
            # 这个规则可以应用于任何 icmp->br 的模式
            fuse_icmp_br: (br (icmp ?mode ?a ?b) ?t ?f) => (br_cmp ?mode ?a ?b ?t ?f)
            
            # 规则2: 将 mul->add->icmp 融合为 mul_add_icmp
            # (icmp mode (+ (* a b) c) d) => (mul_add_icmp mode a b c d)
            # 这个规则只能应用于有 mul 的情况
            fuse_mul_add_icmp: (icmp ?mode (+ (* ?a ?b) ?c) ?d) => (mul_add_icmp ?mode ?a ?b ?c ?d)
        "#;
        
        let rules = parse_rules(rules_str).unwrap();
        
        // 打印初始状态
        println!("=== 初始 e-graph ===");
        println!("E-class 数量: {}", egraph.number_of_classes());
        
        // 检查哪些节点在环上
        let extractor_before = MinCycleExtractor::new(&egraph, is_back_edge_node);
        let (cost_before, expr_before) = extractor_before.find_best(root);
        println!("\n融合前:");
        println!("  表达式: {}", expr_before);
        println!("  环上节点数: {}", cost_before.cycle_node_count);
        println!("  AST 大小: {}", cost_before.ast_size);
        
        // 检查各个节点是否在环上
        println!("\n环1上的节点:");
        println!("  x 在环上: {}", extractor_before.is_on_cycle(x_id));
        println!("  mul1 在环上: {}", extractor_before.is_on_cycle(mul1));
        println!("  add1 在环上: {}", extractor_before.is_on_cycle(add1));
        println!("  icmp1 在环上: {}", extractor_before.is_on_cycle(icmp1));
        println!("  br1 在环上: {}", extractor_before.is_on_cycle(br1));
        
        println!("\n环2上的节点:");
        println!("  y 在环上: {}", extractor_before.is_on_cycle(y_id));
        println!("  add2_1 在环上: {}", extractor_before.is_on_cycle(add2_1));
        println!("  add2_2 在环上: {}", extractor_before.is_on_cycle(add2_2));
        println!("  icmp2 在环上: {}", extractor_before.is_on_cycle(icmp2));
        println!("  br2 在环上: {}", extractor_before.is_on_cycle(br2));
        
        // 运行等式饱和
        let runner = Runner::default()
            .with_egraph(egraph)
            .with_iter_limit(10)
            .run(&rules);
        
        println!("\n=== 饱和后 ===");
        println!("迭代次数: {}", runner.iterations.len());
        println!("E-class 数量: {}", runner.egraph.number_of_classes());
        
        // 使用环感知提取
        let extractor_after = MinCycleExtractor::new(&runner.egraph, is_back_edge_node);
        let (cost_after, expr_after) = extractor_after.find_best(root);
        
        println!("\n融合后:");
        println!("  表达式: {}", expr_after);
        println!("  环上节点数: {}", cost_after.cycle_node_count);
        println!("  AST 大小: {}", cost_after.ast_size);
        
        // 验证融合确实发生了
        let expr_str = expr_after.to_string();
        println!("\n=== 验证 ===");
        
        // 检查是否包含融合后的操作
        let has_br_cmp = expr_str.contains("br_cmp");
        let has_mul_add_icmp = expr_str.contains("mul_add_icmp");
        println!("包含 br_cmp (icmp+br 融合): {}", has_br_cmp);
        println!("包含 mul_add_icmp (mul+add+icmp 融合): {}", has_mul_add_icmp);
        
        // 验证预期行为：
        // 1. 环1应该使用 mul_add_icmp（减少更多节点）
        // 2. 环2应该使用 br_cmp（它没有 mul，只能用这个）
        println!("\n=== 预期分析 ===");
        println!("环1: mul->add->icmp->br，可以使用 mul_add_icmp (减少2个节点) 或 br_cmp (减少1个节点)");
        println!("环2: add->add->icmp->br，只能使用 br_cmp (减少1个节点)，因为没有 mul");
        
        // 确保环上节点数减少了
        assert!(
            cost_after.cycle_node_count < cost_before.cycle_node_count,
            "融合后环上节点数应该减少: {} < {}",
            cost_after.cycle_node_count, cost_before.cycle_node_count
        );
        
        // 确保两种融合都发生了（因为它们针对不同的模式）
        assert!(has_br_cmp, "应该包含 br_cmp 融合（用于环2）");
        assert!(has_mul_add_icmp, "应该包含 mul_add_icmp 融合（用于环1）");
        
        println!("\n节点数减少: {} -> {} (减少了 {} 个环上节点)", 
                 cost_before.cycle_node_count, 
                 cost_after.cycle_node_count,
                 cost_before.cycle_node_count - cost_after.cycle_node_count);
        
        println!("\n测试通过！");
    }
    
    #[test]
    fn test_two_phase_with_custom_area() {
        // 测试两阶段提取：
        // 第一阶段：最小化环上节点数
        // 第二阶段：对于环外节点，根据 area 选择
        //
        // 场景：
        // - 一个环：a -> b, ctrl_mov(b, a)
        // - 两个等价的环外路径：
        //   - expensive_op (area=100)
        //   - cheap_op (area=10)
        
        let mut egraph = EGraph::<SimpleExprLang, ()>::default();
        
        // 构建环
        let a: RecExpr<SimpleExprLang> = "a".parse().unwrap();
        let a_id = egraph.add_expr(&a);
        
        let b = egraph.add(SimpleExprLang::Add([a_id, a_id])); // b = a + a，在环上
        let _ctrl = egraph.add(SimpleExprLang::CtrlMov([b, a_id])); // 创建环
        
        // 两个等价的环外操作
        let result_expensive = egraph.add(SimpleExprLang::Mul([b, b])); // expensive: mul
        let result_cheap = egraph.add(SimpleExprLang::Add([b, b]));     // cheap: add
        
        // 让它们等价
        egraph.union(result_expensive, result_cheap);
        egraph.rebuild();
        
        // 定义 area 函数：mul 的 area=100，add 的 area=10，其他=1
        let area_fn = |node: &SimpleExprLang| -> usize {
            match node {
                SimpleExprLang::Mul(_) => 100,
                SimpleExprLang::Add(_) => 10,
                _ => 1,
            }
        };
        
        let extractor = MinCycleExtractor::new_with_area(&egraph, is_back_edge_node, area_fn);
        let (cost, expr) = extractor.find_best(result_expensive);
        
        println!("=== 两阶段提取测试 ===");
        println!("表达式: {}", expr);
        println!("cycle_node_count: {}", cost.cycle_node_count);
        println!("off_cycle_area: {}", cost.off_cycle_area);
        println!("ast_size: {}", cost.ast_size);
        
        // 验证：
        // 1. cycle_node_count 应该是 2 (a 和 b 在环上)
        // 2. 应该选择 cheap_op (add)，因为它的 area 更小
        let expr_str = expr.to_string();
        
        // 环外节点是 add（cheap），而不是 mul（expensive）
        // 注意：b 是 (+ a a)，而环外节点是 (+ b b) 或 (* b b)
        // 我们应该看到最外层是 add，而不是 mul
        
        // 检查 off_cycle_area
        // a 在环上（area 不计入）
        // b = (+ a a) 在环上（area 不计入）
        // 最外层选 add（area=10）而不是 mul（area=100）
        
        println!("\n验证：应该选择 add (area=10) 而不是 mul (area=100)");
        
        // off_cycle_area 应该是 10（只有一个环外的 add）
        assert_eq!(cost.off_cycle_area, 10, "应该选择 cheap 的 add (area=10)");
        
        println!("测试通过！");
    }
}
