#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Configuration for the equality saturation runner
 */
typedef struct EggConfig {
  /**
   * Maximum number of iterations
   */
  uint32_t iter_limit;
  /**
   * Maximum number of nodes in the e-graph
   */
  uint32_t node_limit;
  /**
   * Time limit in seconds (0 = no limit)
   */
  uint32_t time_limit_secs;
} EggConfig;

/**
 * Result structure for cycle-aware extraction
 */
typedef struct EggCycleResult {
  /**
   * The optimized expression as a string (owned, must be freed)
   */
  char *result_expr;
  /**
   * Error message if any (owned, must be freed), null if success
   */
  char *error_msg;
  /**
   * Number of iterations performed
   */
  uint32_t iterations;
  /**
   * Final e-graph size (number of e-classes)
   */
  uint32_t egraph_size;
  /**
   * Whether saturation was reached
   */
  bool saturated;
  /**
   * Number of nodes on cycles in the extracted expression
   */
  uint32_t cycle_node_count;
  /**
   * Total area of nodes NOT on cycles (for phase 2 optimization)
   */
  uint32_t off_cycle_area;
  /**
   * Total AST size of the extracted expression
   */
  uint32_t ast_size;
} EggCycleResult;

/**
 * Type for the area callback function
 *
 * The callback receives the operation name as a C string and returns the area cost.
 * This allows C++ code to provide custom area costs for different operations.
 */
typedef uint32_t (*AreaCallback)(const char *op_name);

/**
 * Create a default configuration
 */
struct EggConfig egg_config_default(void);

/**
 * Free a string allocated by the Rust library
 */
void egg_string_free(char *s);

/**
 * Get the version of the egg-bridge library
 */
const char *egg_version(void);

/**
 * Run equality saturation with cycle-aware extraction
 *
 * This extraction method uses two-phase optimization:
 * - Phase 1: Minimize the number of nodes on cycles (critical path)
 * - Phase 2: Minimize total area for nodes NOT on cycles
 *
 * Cost semantics:
 * - ctrl_mov operations are treated as backward edges (create cycles)
 * - Only nodes ON cycles contribute to cycle_node_count
 * - Only nodes NOT on cycles contribute to off_cycle_area
 * - Cost comparison: first by cycle_node_count, then by off_cycle_area, then by ast_size
 *
 * # Arguments
 * * `expr_str` - The initial expression as a string (S-expression format)
 * * `rules_str` - The rewrite rules as a string (one rule per line)
 * * `config` - Configuration for the runner
 *
 * # Returns
 * An EggCycleResult containing the optimized expression with cycle information
 */
struct EggCycleResult egg_run_saturation_cycle_aware(const char *expr_str,
                                                     const char *rules_str,
                                                     struct EggConfig config);

/**
 * Run equality saturation with cycle-aware extraction and custom area function
 *
 * This extraction method uses two-phase optimization:
 * - Phase 1: Minimize the number of nodes on cycles (critical path)
 * - Phase 2: Minimize total area for nodes NOT on cycles
 *
 * # Arguments
 * * `expr_str` - The initial expression as a string (S-expression format)
 * * `rules_str` - The rewrite rules as a string (one rule per line)
 * * `config` - Configuration for the runner
 * * `area_callback` - Callback function that returns the area cost for a given operation name
 *
 * # Returns
 * An EggCycleResult containing the optimized expression with cycle information
 */
struct EggCycleResult egg_run_saturation_cycle_aware_with_area(const char *expr_str,
                                                               const char *rules_str,
                                                               struct EggConfig config,
                                                               AreaCallback area_callback);

/**
 * Free the memory allocated for an EggCycleResult
 */
void egg_cycle_result_free(struct EggCycleResult result);
