//! Cycle-aware extraction for e-graphs with control flow edges
//!
//! This module implements a specialized extraction algorithm that:
//! 1. Detects cycles involving `ctrl_mov` (which represents backward edges)
//! 2. Minimizes the number of nodes on cycles
//!
//! In the graph representation:
//! - Normal edges: a -> b means a depends on b
//! - ctrl_mov edges: a -> ctrl_mov -> b actually means b -> a (reverse direction)
//!
//! This creates potential cycles in the actual data flow graph.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use crate::{Analysis, EClass, EGraph, Id, Language, RecExpr};

/// Information about a node's position in cycles
#[derive(Debug, Clone)]
pub struct CycleInfo {
    /// Whether this node is part of any cycle
    pub on_cycle: bool,
    /// The cycle IDs this node belongs to (if any)
    pub cycle_ids: Vec<usize>,
}

impl Default for CycleInfo {
    fn default() -> Self {
        CycleInfo {
            on_cycle: false,
            cycle_ids: Vec::new(),
        }
    }
}

/// A cost function that assigns cost based on whether a node is on a cycle
///
/// Nodes on cycles have cost = 1, nodes not on cycles have cost = 0
/// This encourages the extractor to minimize the number of nodes on cycles
pub struct CycleCostFunction<'a, L: Language, N: Analysis<L>> {
    /// Reference to the e-graph
    pub egraph: &'a EGraph<L, N>,
    /// Map from e-class Id to cycle information
    pub cycle_info: HashMap<Id, CycleInfo>,
    /// Base cost for each node (even if not on cycle)
    pub base_cost: usize,
}

impl<'a, L, N> CycleCostFunction<'a, L, N>
where
    L: Language,
    N: Analysis<L>,
{
    /// Create a new CycleCostFunction by analyzing the e-graph for cycles
    ///
    /// The `is_ctrl_mov` function should return true for e-nodes that represent
    /// backward edges (like ctrl_mov in Neura dialect)
    pub fn new<F>(egraph: &'a EGraph<L, N>, is_ctrl_mov: F) -> Self
    where
        F: Fn(&L) -> bool,
    {
        let cycle_info = Self::detect_cycles(egraph, is_ctrl_mov);
        CycleCostFunction {
            egraph,
            cycle_info,
            base_cost: 1,
        }
    }

    /// Create with a custom base cost
    pub fn with_base_cost<F>(egraph: &'a EGraph<L, N>, is_ctrl_mov: F, base_cost: usize) -> Self
    where
        F: Fn(&L) -> bool,
    {
        let cycle_info = Self::detect_cycles(egraph, is_ctrl_mov);
        CycleCostFunction {
            egraph,
            cycle_info,
            base_cost,
        }
    }

    /// Detect all cycles in the e-graph considering ctrl_mov as reverse edges
    ///
    /// Returns a map from e-class Id to CycleInfo
    fn detect_cycles<F>(egraph: &EGraph<L, N>, is_ctrl_mov: F) -> HashMap<Id, CycleInfo>
    where
        F: Fn(&L) -> bool,
    {
        // Build the actual dependency graph considering ctrl_mov as reverse edges
        // Graph structure: node_id -> (forward_deps, backward_deps_from_ctrl_mov)
        let mut forward_edges: HashMap<Id, HashSet<Id>> = HashMap::new();
        let mut backward_edges: HashMap<Id, HashSet<Id>> = HashMap::new();

        // First pass: collect all edges
        for class in egraph.classes() {
            let class_id = egraph.find(class.id);
            forward_edges.entry(class_id).or_default();
            backward_edges.entry(class_id).or_default();

            for node in class.iter() {
                if is_ctrl_mov(node) {
                    // ctrl_mov(a, b) means b -> a in the actual flow
                    // So if this class contains ctrl_mov with children [a, b],
                    // we add a reverse edge from b to a
                    let children: Vec<Id> = node.children().iter().copied().collect();
                    if children.len() >= 2 {
                        let src = egraph.find(children[0]); // source of ctrl_mov
                        let dst = egraph.find(children[1]); // destination of ctrl_mov
                        // ctrl_mov src -> dst actually means dst -> src in data flow
                        backward_edges.entry(dst).or_default().insert(src);
                    }
                } else {
                    // Normal edge: this node depends on its children
                    for &child in node.children() {
                        let child_id = egraph.find(child);
                        forward_edges.entry(class_id).or_default().insert(child_id);
                    }
                }
            }
        }

        // Build combined graph for cycle detection
        // Edge a -> b exists if:
        //   - a has a forward dependency on b (normal edge)
        //   - OR a has a backward edge to b (from ctrl_mov in the opposite class)
        let mut combined_graph: HashMap<Id, HashSet<Id>> = HashMap::new();
        
        for (&node, forward_deps) in &forward_edges {
            combined_graph.entry(node).or_default().extend(forward_deps);
        }
        
        for (&node, back_deps) in &backward_edges {
            combined_graph.entry(node).or_default().extend(back_deps);
        }

        // Find all cycles using Tarjan's algorithm for strongly connected components
        let sccs = Self::find_strongly_connected_components(&combined_graph);

        // Mark nodes that are on cycles (in SCCs with more than one node, or self-loops)
        let mut cycle_info: HashMap<Id, CycleInfo> = HashMap::new();
        
        for (cycle_id, scc) in sccs.iter().enumerate() {
            let is_cycle = scc.len() > 1 || {
                // Check for self-loop
                scc.len() == 1 && {
                    let node = scc.iter().next().unwrap();
                    combined_graph.get(node).map(|deps| deps.contains(node)).unwrap_or(false)
                }
            };

            if is_cycle {
                for &node in scc {
                    let info = cycle_info.entry(node).or_default();
                    info.on_cycle = true;
                    info.cycle_ids.push(cycle_id);
                }
            }
        }

        // Ensure all nodes have entries
        for class in egraph.classes() {
            let class_id = egraph.find(class.id);
            cycle_info.entry(class_id).or_default();
        }

        cycle_info
    }

    /// Find strongly connected components using Tarjan's algorithm
    fn find_strongly_connected_components(
        graph: &HashMap<Id, HashSet<Id>>,
    ) -> Vec<HashSet<Id>> {
        struct TarjanState {
            index: usize,
            indices: HashMap<Id, usize>,
            lowlinks: HashMap<Id, usize>,
            on_stack: HashSet<Id>,
            stack: Vec<Id>,
            sccs: Vec<HashSet<Id>>,
        }

        fn strongconnect(
            v: Id,
            graph: &HashMap<Id, HashSet<Id>>,
            state: &mut TarjanState,
        ) {
            state.indices.insert(v, state.index);
            state.lowlinks.insert(v, state.index);
            state.index += 1;
            state.stack.push(v);
            state.on_stack.insert(v);

            if let Some(neighbors) = graph.get(&v) {
                for &w in neighbors {
                    if !state.indices.contains_key(&w) {
                        strongconnect(w, graph, state);
                        let lowlink_w = state.lowlinks[&w];
                        let lowlink_v = state.lowlinks[&v];
                        state.lowlinks.insert(v, lowlink_v.min(lowlink_w));
                    } else if state.on_stack.contains(&w) {
                        let lowlink_v = state.lowlinks[&v];
                        let index_w = state.indices[&w];
                        state.lowlinks.insert(v, lowlink_v.min(index_w));
                    }
                }
            }

            if state.lowlinks[&v] == state.indices[&v] {
                let mut scc = HashSet::new();
                loop {
                    let w = state.stack.pop().unwrap();
                    state.on_stack.remove(&w);
                    scc.insert(w);
                    if w == v {
                        break;
                    }
                }
                state.sccs.push(scc);
            }
        }

        let mut state = TarjanState {
            index: 0,
            indices: HashMap::new(),
            lowlinks: HashMap::new(),
            on_stack: HashSet::new(),
            stack: Vec::new(),
            sccs: Vec::new(),
        };

        for &node in graph.keys() {
            if !state.indices.contains_key(&node) {
                strongconnect(node, graph, &mut state);
            }
        }

        state.sccs
    }

    /// Get the cost for an e-class
    pub fn get_cost(&self, id: Id) -> usize {
        let canonical_id = self.egraph.find(id);
        let info = self.cycle_info.get(&canonical_id).cloned().unwrap_or_default();
        
        if info.on_cycle {
            self.base_cost + 1 // Extra cost for being on a cycle
        } else {
            self.base_cost
        }
    }

    /// Check if an e-class is on a cycle
    pub fn is_on_cycle(&self, id: Id) -> bool {
        let canonical_id = self.egraph.find(id);
        self.cycle_info
            .get(&canonical_id)
            .map(|info| info.on_cycle)
            .unwrap_or(false)
    }
}

/// Cycle-aware extractor that minimizes nodes on cycles
pub struct CycleExtractor<'a, L: Language, N: Analysis<L>> {
    egraph: &'a EGraph<L, N>,
    costs: HashMap<Id, (usize, L)>,
    cycle_info: HashMap<Id, CycleInfo>,
}

impl<'a, L, N> CycleExtractor<'a, L, N>
where
    L: Language,
    N: Analysis<L>,
{
    /// Create a new CycleExtractor
    ///
    /// The `is_ctrl_mov` function should return true for e-nodes that represent
    /// backward edges (like ctrl_mov in Neura dialect)
    pub fn new<F>(egraph: &'a EGraph<L, N>, is_ctrl_mov: F) -> Self
    where
        F: Fn(&L) -> bool + Clone,
    {
        let cycle_info = CycleCostFunction::detect_cycles(egraph, is_ctrl_mov.clone());
        let mut extractor = CycleExtractor {
            egraph,
            costs: HashMap::default(),
            cycle_info,
        };
        extractor.find_costs(is_ctrl_mov);
        extractor
    }

    /// Find the best (lowest cycle-cost) represented RecExpr in the given e-class
    pub fn find_best(&self, eclass: Id) -> (usize, RecExpr<L>) {
        let (cost, root) = self.costs[&self.egraph.find(eclass)].clone();
        let expr = root.build_recexpr(|id| self.find_best_node(id).clone());
        (cost, expr)
    }

    /// Find the best e-node in the given e-class
    pub fn find_best_node(&self, eclass: Id) -> &L {
        &self.costs[&self.egraph.find(eclass)].1
    }

    /// Find the cost of the term that would be extracted from this e-class
    pub fn find_best_cost(&self, eclass: Id) -> usize {
        let (cost, _) = &self.costs[&self.egraph.find(eclass)];
        *cost
    }

    /// Check if an e-class is on a cycle
    pub fn is_on_cycle(&self, id: Id) -> bool {
        let canonical_id = self.egraph.find(id);
        self.cycle_info
            .get(&canonical_id)
            .map(|info| info.on_cycle)
            .unwrap_or(false)
    }

    /// Get all e-classes that are on cycles
    pub fn get_cycle_nodes(&self) -> Vec<Id> {
        self.cycle_info
            .iter()
            .filter(|(_, info)| info.on_cycle)
            .map(|(&id, _)| id)
            .collect()
    }

    fn node_total_cost<F>(&self, node: &L, is_ctrl_mov: &F) -> Option<usize>
    where
        F: Fn(&L) -> bool,
    {
        let eg = self.egraph;
        let has_cost = |id| self.costs.contains_key(&eg.find(id));
        
        if node.all(has_cost) {
            let costs = &self.costs;
            let child_cost_sum: usize = node.fold(0usize, |sum, id| {
                sum.saturating_add(costs[&eg.find(id)].0)
            });
            
            // Node cost: 1 if on cycle, 0 otherwise
            // But we need to check if THIS node being selected affects cycles
            // For ctrl_mov nodes, we give them a cost based on their role
            let node_cost = if is_ctrl_mov(node) {
                // ctrl_mov nodes contribute to cycles, give them higher cost
                2
            } else {
                1 // Base cost
            };
            
            Some(child_cost_sum.saturating_add(node_cost))
        } else {
            None
        }
    }

    fn find_costs<F>(&mut self, is_ctrl_mov: F)
    where
        F: Fn(&L) -> bool,
    {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for class in self.egraph.classes() {
                let pass = self.make_pass(class, &is_ctrl_mov);
                match (self.costs.get(&class.id), pass) {
                    (None, Some(new)) => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    (Some(old), Some(new)) if new.0 < old.0 => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    _ => (),
                }
            }
        }

        for class in self.egraph.classes() {
            if !self.costs.contains_key(&class.id) {
                log::warn!(
                    "Failed to compute cost for eclass {}: {:?}",
                    class.id,
                    class.nodes
                )
            }
        }
    }

    fn make_pass<F>(&self, eclass: &EClass<L, N::Data>, is_ctrl_mov: &F) -> Option<(usize, L)>
    where
        F: Fn(&L) -> bool,
    {
        fn cmp_opt(a: &Option<usize>, b: &Option<usize>) -> std::cmp::Ordering {
            use std::cmp::Ordering;
            match (a, b) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Greater,
                (Some(_), None) => Ordering::Less,
                (Some(a), Some(b)) => a.cmp(b),
            }
        }

        let (cost, node) = eclass
            .iter()
            .map(|n| (self.node_total_cost(n, is_ctrl_mov), n))
            .min_by(|a, b| cmp_opt(&a.0, &b.0))
            .unwrap_or_else(|| panic!("Can't extract, eclass is empty: {:#?}", eclass));
        cost.map(|c| (c, node.clone()))
    }
}

/// Advanced cycle-aware extractor that considers the actual cycle structure
/// and tries to minimize the total count of nodes on cycles
/// 
/// Key design choices:
/// - Only nodes that are on cycles have cost = 1
/// - Nodes NOT on cycles have cost = 0
/// - Cycles are created by ctrl_mov backward edges
/// - Cost aggregation: cost = sum(child_costs) + self_cost
/// - Goal: minimize the total number of nodes on cycles
pub struct MinCycleExtractor<'a, L: Language, N: Analysis<L>> {
    egraph: &'a EGraph<L, N>,
    costs: HashMap<Id, (CycleCost, L)>,
    cycle_info: HashMap<Id, CycleInfo>,
}

/// Cost structure that tracks nodes on cycles and total size
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleCost {
    /// Number of nodes on cycles (the key metric to minimize)
    pub cycle_node_count: usize,
    /// Total AST size (for tie-breaking)
    pub ast_size: usize,
}

impl CycleCost {
    /// Create a new CycleCost with specified cycle_node_count and ast_size
    pub fn new(cycle_node_count: usize, ast_size: usize) -> Self {
        CycleCost { cycle_node_count, ast_size }
    }

    /// Create a zero-cost CycleCost
    pub fn zero() -> Self {
        CycleCost { cycle_node_count: 0, ast_size: 0 }
    }
}

impl PartialOrd for CycleCost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CycleCost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Primary: minimize cycle node count
        // Secondary: minimize AST size (for tie-breaking)
        match self.cycle_node_count.cmp(&other.cycle_node_count) {
            std::cmp::Ordering::Equal => self.ast_size.cmp(&other.ast_size),
            other => other,
        }
    }
}

impl<'a, L, N> MinCycleExtractor<'a, L, N>
where
    L: Language,
    N: Analysis<L>,
{
    /// Create a new MinCycleExtractor
    ///
    /// Arguments:
    /// - `egraph`: The e-graph to extract from
    /// - `is_back_edge`: Returns true for nodes that represent backward edges (ctrl_mov)
    ///                   These backward edges are used for cycle detection.
    pub fn new<F>(egraph: &'a EGraph<L, N>, is_back_edge: F) -> Self
    where
        F: Fn(&L) -> bool + Clone,
    {
        let cycle_info = CycleCostFunction::detect_cycles(egraph, is_back_edge);
        let mut extractor = MinCycleExtractor {
            egraph,
            costs: HashMap::default(),
            cycle_info,
        };
        extractor.find_costs();
        extractor
    }

    /// Find the best (minimum cycle nodes) represented RecExpr in the given e-class
    pub fn find_best(&self, eclass: Id) -> (CycleCost, RecExpr<L>) {
        let (cost, root) = self.costs[&self.egraph.find(eclass)].clone();
        let expr = root.build_recexpr(|id| self.find_best_node(id).clone());
        (cost, expr)
    }

    /// Find the best e-node in the given e-class
    pub fn find_best_node(&self, eclass: Id) -> &L {
        &self.costs[&self.egraph.find(eclass)].1
    }

    /// Find the cost of the term that would be extracted from this e-class
    pub fn find_best_cost(&self, eclass: Id) -> CycleCost {
        let (cost, _) = &self.costs[&self.egraph.find(eclass)];
        cost.clone()
    }

    /// Check if an e-class is on a cycle
    pub fn is_on_cycle(&self, id: Id) -> bool {
        let canonical_id = self.egraph.find(id);
        self.cycle_info
            .get(&canonical_id)
            .map(|info| info.on_cycle)
            .unwrap_or(false)
    }

    /// Calculate cost for a node
    /// - Node on cycle: cost = sum(child_costs) + 1
    /// - Node not on cycle: cost = sum(child_costs) + 0
    fn node_total_cost(&self, node: &L, node_eclass: Id) -> Option<CycleCost>
    {
        let eg = self.egraph;
        let has_cost = |id| self.costs.contains_key(&eg.find(id));
        
        if node.all(has_cost) {
            let costs = &self.costs;
            
            // Sum child costs
            let child_cost_sum = node.fold(CycleCost::zero(), |acc, id| {
                let child_cost = &costs[&eg.find(id)].0;
                CycleCost {
                    cycle_node_count: acc.cycle_node_count + child_cost.cycle_node_count,
                    ast_size: acc.ast_size + child_cost.ast_size,
                }
            });
            
            // Check if this e-class is on a cycle
            let canonical_id = eg.find(node_eclass);
            let on_cycle = self.cycle_info
                .get(&canonical_id)
                .map(|info| info.on_cycle)
                .unwrap_or(false);
            
            // Nodes on cycles add 1 to cycle_node_count
            // All nodes add 1 to ast_size
            let self_cycle_cost = if on_cycle { 1 } else { 0 };
            
            Some(CycleCost {
                cycle_node_count: child_cost_sum.cycle_node_count + self_cycle_cost,
                ast_size: child_cost_sum.ast_size + 1,
            })
        } else {
            None
        }
    }

    fn find_costs(&mut self)
    {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for class in self.egraph.classes() {
                let pass = self.make_pass(class);
                match (self.costs.get(&class.id), pass) {
                    (None, Some(new)) => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    (Some(old), Some(new)) if new.0 < old.0 => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    _ => (),
                }
            }
        }

        for class in self.egraph.classes() {
            if !self.costs.contains_key(&class.id) {
                log::warn!(
                    "Failed to compute cost for eclass {}: {:?}",
                    class.id,
                    class.nodes
                )
            }
        }
    }

    fn make_pass(&self, eclass: &EClass<L, N::Data>) -> Option<(CycleCost, L)>
    {
        fn cmp_opt(a: &Option<CycleCost>, b: &Option<CycleCost>) -> std::cmp::Ordering {
            use std::cmp::Ordering;
            match (a, b) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Greater,
                (Some(_), None) => Ordering::Less,
                (Some(a), Some(b)) => a.cmp(b),
            }
        }

        let (cost, node) = eclass
            .iter()
            .map(|n| (self.node_total_cost(n, eclass.id), n))
            .min_by(|a, b| cmp_opt(&a.0, &b.0))
            .unwrap_or_else(|| panic!("Can't extract, eclass is empty: {:#?}", eclass));
        cost.map(|c| (c, node.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SymbolLang;

    /// Check if node is a backward edge (ctrl_mov)
    fn is_back_edge(node: &SymbolLang) -> bool {
        node.op.as_str() == "ctrl_mov"
    }

    #[test]
    fn test_simple_cycle_detection() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        
        // Create a simple graph: a -> b -> c -> (ctrl_mov back to a)
        // This should form a cycle: a -> b -> c -> a
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::new("op", vec![a]));
        let c = egraph.add(SymbolLang::new("op2", vec![b]));
        // ctrl_mov c -> a means a -> c in the actual flow, completing the cycle
        let _ctrl = egraph.add(SymbolLang::new("ctrl_mov", vec![c, a]));
        
        egraph.rebuild();
        
        let extractor = CycleExtractor::new(&egraph, is_back_edge);
        
        let cycle_nodes = extractor.get_cycle_nodes();
        println!("Cycle nodes: {:?}", cycle_nodes);
    }

    #[test]
    fn test_no_cycle() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        
        // Create a simple DAG without cycles
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let _c = egraph.add(SymbolLang::new("add", vec![a, b]));
        
        egraph.rebuild();
        
        let extractor = CycleExtractor::new(&egraph, is_back_edge);
        
        // No nodes should be on cycles
        let cycle_nodes = extractor.get_cycle_nodes();
        assert!(cycle_nodes.is_empty(), "Expected no cycle nodes, got: {:?}", cycle_nodes);
    }

    #[test]
    fn test_min_cycle_no_cycle_graph() {
        // Test a graph with NO cycles - all nodes should have cycle_node_count = 0
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        
        // Create expression: (add (mul a b) c)
        // This is a pure DAG with no ctrl_mov, so no cycles
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let c = egraph.add(SymbolLang::leaf("c"));
        let mul = egraph.add(SymbolLang::new("mul", vec![a, b]));
        let add = egraph.add(SymbolLang::new("add", vec![mul, c]));
        
        egraph.rebuild();
        
        let extractor = MinCycleExtractor::new(&egraph, is_back_edge);
        let (cost, expr) = extractor.find_best(add);
        
        println!("Best expression: {}", expr);
        println!("Cost: cycle_node_count={}, ast_size={}", cost.cycle_node_count, cost.ast_size);
        
        // No cycles, so cycle_node_count should be 0
        assert_eq!(cost.cycle_node_count, 0);
        // AST size should be 5: a, b, c, mul, add
        assert_eq!(cost.ast_size, 5);
    }

    #[test]
    fn test_min_cycle_with_cycle() {
        // Test a graph WITH cycles - nodes on cycle should have cost
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        
        // Create a cycle: a -> b -> c, ctrl_mov(c, a) means a -> c
        // So we have: a -> b -> c -> a (cycle of 3 nodes)
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::new("op1", vec![a]));
        let c = egraph.add(SymbolLang::new("op2", vec![b]));
        let ctrl = egraph.add(SymbolLang::new("ctrl_mov", vec![c, a]));
        
        egraph.rebuild();
        
        let extractor = MinCycleExtractor::new(&egraph, is_back_edge);
        
        // Check which nodes are on cycle
        println!("a on cycle: {}", extractor.is_on_cycle(a));
        println!("b on cycle: {}", extractor.is_on_cycle(b));
        println!("c on cycle: {}", extractor.is_on_cycle(c));
        println!("ctrl on cycle: {}", extractor.is_on_cycle(ctrl));
        
        // The cycle should include a, b, c (and possibly ctrl)
        let cost_a = extractor.find_best_cost(a);
        let cost_b = extractor.find_best_cost(b);
        let cost_c = extractor.find_best_cost(c);
        
        println!("Cost a: {:?}", cost_a);
        println!("Cost b: {:?}", cost_b);
        println!("Cost c: {:?}", cost_c);
        
        // Nodes on cycle should have non-zero cycle_node_count
        // a is leaf on cycle: cycle_node_count = 1
        // b depends on a, b is on cycle: cycle_node_count = 1 + 1 = 2
        // c depends on b, c is on cycle: cycle_node_count = 2 + 1 = 3
    }

    #[test]
    fn test_choose_path_with_fewer_cycle_nodes() {
        // Test that extraction prefers paths with fewer cycle nodes
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        
        // Create two equivalent expressions:
        // Path 1: Through a cycle (a -> b -> c with ctrl_mov making it cyclic)
        // Path 2: Through a non-cycle path (x -> y)
        
        // Non-cycle path
        let x = egraph.add(SymbolLang::leaf("x"));
        let y = egraph.add(SymbolLang::new("f", vec![x]));
        
        // Cycle path
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::new("g", vec![a]));
        let c = egraph.add(SymbolLang::new("h", vec![b]));
        let _ctrl = egraph.add(SymbolLang::new("ctrl_mov", vec![c, a]));  // creates cycle
        
        // Another node that references c
        let result_cyclic = egraph.add(SymbolLang::new("result", vec![c]));
        
        // Union the non-cyclic and cyclic versions
        // First create a result for y
        let result_non_cyclic = egraph.add(SymbolLang::new("result", vec![y]));
        
        egraph.union(result_cyclic, result_non_cyclic);
        egraph.rebuild();
        
        let extractor = MinCycleExtractor::new(&egraph, is_back_edge);
        let (cost, expr) = extractor.find_best(result_cyclic);
        
        println!("Best expression: {}", expr);
        println!("Cost: cycle_node_count={}, ast_size={}", cost.cycle_node_count, cost.ast_size);
        
        // Should prefer the non-cyclic path (x -> y -> result)
        // cycle_node_count should be 0 (no nodes on cycle in chosen path)
        assert_eq!(cost.cycle_node_count, 0, "Should choose non-cyclic path");
    }

    #[test]
    fn test_all_costs_for_simple_dag() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        
        // Create: add(mul(a, b), c)
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let c = egraph.add(SymbolLang::leaf("c"));
        let mul = egraph.add(SymbolLang::new("mul", vec![a, b]));
        let add = egraph.add(SymbolLang::new("add", vec![mul, c]));
        
        egraph.rebuild();
        
        let extractor = MinCycleExtractor::new(&egraph, is_back_edge);
        
        // Check individual costs - none should be on cycle
        let cost_a = extractor.find_best_cost(a);
        let cost_b = extractor.find_best_cost(b);
        let cost_c = extractor.find_best_cost(c);
        let cost_mul = extractor.find_best_cost(mul);
        let cost_add = extractor.find_best_cost(add);
        
        println!("Cost a: {:?}", cost_a);
        println!("Cost b: {:?}", cost_b);
        println!("Cost c: {:?}", cost_c);
        println!("Cost mul: {:?}", cost_mul);
        println!("Cost add: {:?}", cost_add);
        
        // All cycle_node_count should be 0 (no cycles in DAG)
        assert_eq!(cost_a.cycle_node_count, 0);
        assert_eq!(cost_b.cycle_node_count, 0);
        assert_eq!(cost_c.cycle_node_count, 0);
        assert_eq!(cost_mul.cycle_node_count, 0);
        assert_eq!(cost_add.cycle_node_count, 0);
        
        // AST sizes should accumulate
        assert_eq!(cost_a.ast_size, 1);
        assert_eq!(cost_b.ast_size, 1);
        assert_eq!(cost_c.ast_size, 1);
        assert_eq!(cost_mul.ast_size, 3); // a + b + mul
        assert_eq!(cost_add.ast_size, 5); // mul(3) + c(1) + add(1)
    }

    #[test]
    fn test_cycle_node_count_accumulation() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        
        // Create a cycle and verify cycle_node_count accumulates correctly
        // a -> b, ctrl_mov(b, a) creates cycle a <-> b
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::new("op", vec![a]));
        let _ctrl = egraph.add(SymbolLang::new("ctrl_mov", vec![b, a]));
        
        // c depends on b but c is NOT on the cycle
        // (c is reachable from cycle but doesn't form part of cycle)
        // Actually in this graph structure, c won't be on cycle because 
        // there's no back edge TO c
        let c = egraph.add(SymbolLang::new("outside", vec![b]));
        
        egraph.rebuild();
        
        let extractor = MinCycleExtractor::new(&egraph, is_back_edge);
        
        println!("a on cycle: {}", extractor.is_on_cycle(a));
        println!("b on cycle: {}", extractor.is_on_cycle(b));
        println!("c on cycle: {}", extractor.is_on_cycle(c));
        
        let cost_a = extractor.find_best_cost(a);
        let cost_b = extractor.find_best_cost(b);
        let cost_c = extractor.find_best_cost(c);
        
        println!("Cost a: {:?}", cost_a);
        println!("Cost b: {:?}", cost_b);
        println!("Cost c: {:?}", cost_c);
        
        // a and b are on cycle
        // c is NOT on cycle (it just depends on a node that is on cycle)
        // 
        // cost_a: cycle_node_count = 1 (a is on cycle)
        // cost_b: cycle_node_count = cost_a + 1 = 2 (b is on cycle)
        // cost_c: cycle_node_count = cost_b + 0 = 2 (c is NOT on cycle, doesn't add)
        
        assert!(extractor.is_on_cycle(a), "a should be on cycle");
        assert!(extractor.is_on_cycle(b), "b should be on cycle");
        assert!(!extractor.is_on_cycle(c), "c should NOT be on cycle");
        
        assert_eq!(cost_a.cycle_node_count, 1);
        assert_eq!(cost_b.cycle_node_count, 2);
        assert_eq!(cost_c.cycle_node_count, 2); // inherits from b, but c itself doesn't add
    }
}

