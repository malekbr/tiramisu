#include <isl/ctx.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/constraint.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string>
#include <algorithm>

namespace tiramisu
{
std::map<std::string, computation *> computations_list;
bool global::auto_data_mapping;
primitive_t global::loop_iterator_type = p_int32;

// Used for the generation of new variable names.
int id_counter = 0;

const var computation::root = var("root");

static int next_dim_name = 0;

/**
 * Retrieve the access function of the ISL AST leaf node (which represents a
 * computation). Store the access in computation->access.
 */
isl_ast_node *for_code_generator_after_for(
        isl_ast_node *node, isl_ast_build *build, void *user);

std::string generate_new_variable_name();

tiramisu::expr traverse_expr_and_replace_non_affine_accesses(tiramisu::computation *comp,
                                                             const tiramisu::expr &exp);

tiramisu::expr tiramisu_expr_from_isl_ast_expr(isl_ast_expr *isl_expr);

/**
  * Add a dimension to the range of a map in the specified position.
  * Assume that the name of the new dimension is equal to the name of the corresponding
  * dimension in the domain of the map.
  * Add a constraint that indicates that the added dim is equal to a constant.
  */
isl_map *isl_map_add_dim_and_eq_constraint(isl_map *map, int dim_pos, int constant);

/**
 * Create an equality constraint and add it to the schedule \p sched.
 * Edit the schedule as follows: assuming that y and y' are the input
 * and output dimensions of sched in dimensions \p dim0.
 * This function function add the constraint:
 *   in_dim_coefficient*y = out_dim_coefficient*y' + const_conefficient;
 */
isl_map *add_eq_to_schedule_map(int dim0, int in_dim_coefficient, int out_dim_coefficient,
                                int const_conefficient, isl_map *sched);

/**
 * Create an inequality constraint and add it to the schedule \p sched
 * of the duplicate computation that has \p duplicate_ID as an ID.
 * Edit the schedule as follows: assuming that y and y' are the input
 * and output dimensions of sched in dimensions \p dim0.
 * This function function add the constraint:
 *   in_dim_coefficient*y <= out_dim_coefficient*y' + const_conefficient;
 */
isl_map *add_ineq_to_schedule_map(int duplicate_ID, int dim0, int in_dim_coefficient,
                                  int out_dim_coefficient, int const_conefficient, isl_map *sched);


/**
  * Add a buffer to the function.
  */
void function::add_buffer(std::pair < std::string, tiramisu::buffer * > buf)
{
        assert(!buf.first.empty() && ("Empty buffer name."));
        assert((buf.second != NULL) && ("Empty buffer."));

        this->buffers_list.insert(buf);
}

/**
 * Construct a function with the name \p name.
 */
function::function(std::string name) {
        assert(!name.empty() && ("Empty function name"));

        this->name = name;
        halide_stmt = Halide::Internal::Stmt();
        ast = NULL;
        context_set = NULL;
        use_low_level_scheduling_commands = false;
        _needs_rank_call = false;

        // Allocate an ISL context.  This ISL context will be used by
        // the ISL library calls within Tiramisu.
        ctx = isl_ctx_alloc();
};

/**
  * Get the arguments of the function.
  */
// @{
const std::vector<tiramisu::buffer *> &function::get_arguments() const {
    return function_arguments;
}
// @}

isl_union_map *tiramisu::function::compute_dep_graph() {
        DEBUG_FCT_NAME(3);
        DEBUG_INDENT(4);

        isl_union_map *result = NULL;

        for (const auto &consumer : this->get_computations()) {
            DEBUG(3, tiramisu::str_dump("Computing the dependences involving the computation " +
                                        consumer->get_name() + "."));
            DEBUG(3, tiramisu::str_dump("Computing the accesses of the computation."));

            isl_union_map *accesses_union_map = NULL;
            std::vector < isl_map * > accesses_vector;
            generator::get_rhs_accesses(this, consumer, accesses_vector, false);

            DEBUG(3, tiramisu::str_dump("Vector of accesses computed."));

            if (!accesses_vector.empty()) {
                // Create a union map of the accesses to the producer.
                if (accesses_union_map == NULL) {
                    isl_space *space = isl_map_get_space(accesses_vector[0]);
                    assert(space != NULL);
                    accesses_union_map = isl_union_map_empty(space);
                }

                for (size_t i = 0; i < accesses_vector.size(); ++i) {
                    isl_map *reverse_access = isl_map_reverse(accesses_vector[i]);
                    accesses_union_map = isl_union_map_union(isl_union_map_from_map(reverse_access),
                                                             accesses_union_map);
                }

                //accesses_union_map = isl_union_map_intersect_range(accesses_union_map, isl_union_set_from_set(isl_set_copy(consumer->get_iteration_domain())));
                //accesses_union_map = isl_union_map_intersect_domain(accesses_union_map, isl_union_set_from_set(isl_set_copy(consumer->get_iteration_domain())));

                DEBUG(3, tiramisu::str_dump("Accesses after filtering."));
                DEBUG(3, tiramisu::str_dump(isl_union_map_to_str(accesses_union_map)));

                if (result == NULL) {
                    result = isl_union_map_copy(accesses_union_map);
                    isl_union_map_free(accesses_union_map);
                } else {
                    result = isl_union_map_union(result, accesses_union_map);
                }
            }
        }

        DEBUG(3, tiramisu::str_dump("Dep graph: "));
        if (result != NULL)
        {
            DEBUG(3, tiramisu::str_dump(isl_union_map_to_str(result)));
        }
        else
        {
            DEBUG(3, tiramisu::str_dump("Null."));
        }

        DEBUG_INDENT(-4);
        DEBUG(3, tiramisu::str_dump("End of function"));

        return result;
    }

// TODO: get_live_in_computations() does not consider the case of "maybe"
// live-out (non-affine control flow, ...).
std::vector<tiramisu::computation *> tiramisu::function::get_live_in_computations()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert((this->get_computations().size() > 0) &&
           "The function should have at least one computation.");

    std::vector < tiramisu::computation * > first;
    isl_union_map *deps = this->compute_dep_graph();

    if (deps != NULL) {
        if (isl_union_map_is_empty(deps) == isl_bool_false) {
            // The domains and the ranges of the dependences
            isl_union_set *domains = isl_union_map_domain(isl_union_map_copy(deps));
            isl_union_set *ranges = isl_union_map_range(isl_union_map_copy(deps));

            DEBUG(3, tiramisu::str_dump("Ranges of the dependence graph.", isl_union_set_to_str(ranges)));
            DEBUG(3, tiramisu::str_dump("Domains of the dependence graph.", isl_union_set_to_str(domains)));

            /** In a dependence graph, since dependences create a chain (i.e., the end of
             *  a dependence is the beginning of the following), then each range of
             *  a dependence has a set domains that correspond to it (i.e., that their
             *  union is equal to it).  If a domain exists but does not have ranges that
             *  are equal to it, then that domain is the first domain.
             *
             *  To compute those domains that do not have corresponding ranges, we
             *  compute (domains - ranges).
             *
             *  These domains that do not have a corresponding range (i.e., are not
             *  produced by previous computations) and that are not defined (i.e., do
             *  not have any expression) are live-in.
             */
            isl_union_set *first_domains = isl_union_set_subtract(domains, ranges);
            DEBUG(3, tiramisu::str_dump("Domains - Ranges :", isl_union_set_to_str(first_domains)));

            if (isl_union_set_is_empty(first_domains) == isl_bool_false) {
                for (const auto &c : this->body) {
                    isl_space *sp = isl_set_get_space(c->get_iteration_domain());
                    isl_set *s = isl_set_universe(sp);
                    isl_union_set *intersect =
                            isl_union_set_intersect(isl_union_set_from_set(s),
                                                    isl_union_set_copy(first_domains));

                    if ((isl_union_set_is_empty(intersect) == isl_bool_false) &&
                        (c->get_expr().is_defined() == false))
                    {
                        first.push_back(c);
                    }
                    isl_union_set_free(intersect);
                }

                DEBUG(3, tiramisu::str_dump("First computations:"));
                for (const auto &c : first) {
                    DEBUG(3, tiramisu::str_dump(c->get_name() + " "));
                }
            } else {
                // If the difference between domains and ranges is empty, then
                // all the computations of the program are recursive (assuming
                // that the set of dependences is not empty).
                first = this->body;
            }

            isl_union_set_free(first_domains);
        } else {
            // If the program does not have any dependence, then
            // all the computations should be considered as the first
            // computations.
            first = this->body;
        }

        isl_union_map_free(deps);
    }

    DEBUG(3, tiramisu::str_dump("Removing inline computations."));
    std::vector<computation* > result;
    for (computation * c: first)
        if (! c->is_inline_computation())
            result.push_back(c);

    DEBUG_INDENT(-4);

    return result;
}

// TODO: get_live_out_computations() does not consider the case of "maybe"
// live-out (non-affine control flow, ...).
std::vector<tiramisu::computation *> tiramisu::function::get_live_out_computations()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert((this->get_computations().size() > 0) &&
           "The function should have at least one computation.");

    std::vector<tiramisu::computation *> first;
    isl_union_map *deps = this->compute_dep_graph();

    if (deps != NULL)
    {
        if (isl_union_map_is_empty(deps) == isl_bool_false)
        {
            // The domains and the ranges of the dependences
            isl_union_set *domains = isl_union_map_domain(isl_union_map_copy(deps));
            isl_union_set *ranges = isl_union_map_range(isl_union_map_copy(deps));

            DEBUG(3, tiramisu::str_dump("Ranges of the dependence graph.", isl_union_set_to_str(ranges)));
            DEBUG(3, tiramisu::str_dump("Domains of the dependence graph.", isl_union_set_to_str(domains)));

            /** In a dependence graph, since dependences create a chain (i.e., the end of
             *  a dependence is the beginning of the following), then each range of
             *  a dependence has a set domains that correspond to it (i.e., that their
             *  union is equal to it).  If a range exists but does not have domains that
             *  are equal to it, then that range is the last range.
             *
             *  To compute those ranges that do not have corresponding domains, we
             *  compute (ranges - domains).
             */
            isl_union_set *last_ranges = isl_union_set_subtract(ranges, domains);
            DEBUG(3, tiramisu::str_dump("Ranges - Domains :", isl_union_set_to_str(last_ranges)));

            if (isl_union_set_is_empty(last_ranges) == isl_bool_false)
            {
                for (const auto &c : this->body)
                {
                    isl_space *sp = isl_set_get_space(c->get_iteration_domain());
                    isl_set *s = isl_set_universe(sp);
                    isl_union_set *intersect =
                        isl_union_set_intersect(isl_union_set_from_set(s),
                                                isl_union_set_copy(last_ranges));

                    if (isl_union_set_is_empty(intersect) == isl_bool_false)
                    {
                        first.push_back(c);
                    }
                    isl_union_set_free(intersect);
                }

                DEBUG(3, tiramisu::str_dump("Last computations:"));
                for (const auto &c : first)
                {
                    DEBUG(3, tiramisu::str_dump(c->get_name() + " "));
                }
            }
            else
            {
                // If the difference between ranges and domains is empty, then
                // all the computations of the program are recursive (assuming
                // that the set of dependences is not empty).
                first = this->body;
            }

            isl_union_set_free(last_ranges);
        }
        else
        {
            // If the program does not have any dependence, then
            // all the computations should be considered as the last
            // computations.
            first = this->body;
        }
        isl_union_map_free(deps);
    }
    else
    {
        // If the program does not have any dependence, then
        // all the computations should be considered as the last
        // computations.
        first = this->body;
    }

    DEBUG(3, tiramisu::str_dump("Removing inline computations."));
    std::vector<computation* > result;
    for (computation * c: first)
        if (! c->is_inline_computation())
            result.push_back(c);

    assert((result.size() > 0) && "The function should have at least one last computation.");

    DEBUG_INDENT(-4);


    return result;
}


isl_set *tiramisu::computation::get_iteration_domains_of_all_definitions()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::string name = this->get_name();
    assert(name.size() > 0);
    isl_set *result = NULL;
    isl_space *space = NULL;

    assert(isl_set_is_empty(this->get_iteration_domain()) == isl_bool_false);
    space = isl_set_get_space(this->get_iteration_domain());
    assert(space != NULL);
    result = isl_set_empty(space);

    std::vector<tiramisu::computation *> computations =
        this->get_function()->get_computation_by_name(name);

    for (auto c : computations)
    {
        if (c->should_schedule_this_computation())
        {
            isl_set *c_iter_space = isl_set_copy(c->get_iteration_domain());
            result = isl_set_union(c_iter_space, result);
        }
    }

    DEBUG_INDENT(-4);

    return result;
}

bool tiramisu::computation::has_multiple_definitions()
{
    bool is_update = false;

    std::string name = this->get_name();
    assert(name.size() > 0);

    std::vector<tiramisu::computation *> computations =
        this->get_function()->get_computation_by_name(name);

    if (computations.size() > 1)
    {
        is_update = true;
    }

    if (this->get_updates().size() > 1)
	    is_update = true;

    if (this->get_first_definition() != NULL)
        if (this->get_first_definition()->get_updates().size() > 1)
	    is_update = true;

    return is_update;
}

void tiramisu::function::compute_bounds()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_union_map *Dep = this->compute_dep_graph();
    DEBUG(3, tiramisu::str_dump("Dependences computed."));
    isl_union_map *Reverse = isl_union_map_reverse(isl_union_map_copy(Dep));
    DEBUG(3, tiramisu::str_dump("Reverse of dependences:", isl_union_map_to_str(Reverse)));
    // Compute the vector of the last computations in the dependence graph
    // (i.e., the computations that do not have any consumer).
    std::vector<tiramisu::computation *> first = this->get_live_out_computations();

    assert(first.size() > 0);

    isl_union_set *Domains = NULL;
    Domains = isl_union_set_empty(isl_set_get_space(first[0]->get_iteration_domain()));
    for (auto c : first)
    {
        Domains = isl_union_set_union(Domains,
                                      isl_union_set_from_set(isl_set_copy(c->get_iteration_domain())));
    }
    DEBUG(3, tiramisu::str_dump("The domains of the last computations are:",
                                isl_union_set_to_str(Domains)));

    // Compute "Producers", the union of the iteration domains of the computations
    // that computed "last".
    isl_union_set *Producers = isl_union_set_apply(isl_union_set_copy(Domains),
                               isl_union_map_copy(Reverse));
    DEBUG(3, tiramisu::str_dump("The producers of the last computations are:",
                                isl_union_set_to_str(Producers)));

    // If the graph of dependences has recursive dependences, then the intersection of
    // the old producers and the new producers will not be empty (i.e., the old producer and the new producer
    // are the same).
    // In this case, we should subtract the common domain so that in the next iterations of the
    // the algorithm we do not get the same computation again and again (since we have a recursive dependence).
    // This is equivalent to removing the recursive dependence (we remove its image instead of removing it).
    isl_union_set *old_Producers = isl_union_set_copy(Domains);
    isl_union_set *intersection = isl_union_set_intersect(old_Producers, isl_union_set_copy(Producers));
    if (isl_union_set_is_empty(intersection) == isl_bool_false)
    {
        isl_union_set *common_computations = isl_union_set_universe(intersection);
        Producers = isl_union_set_subtract(Producers, common_computations);
        DEBUG(3, tiramisu::str_dump("After eliminating the effect of recursive dependences.",
                                    isl_union_set_to_str(Producers)));
    }


    // Propagation of bounds
    DEBUG(3, tiramisu::str_dump("Propagating the bounds over all computations."));
    DEBUG_INDENT(4);
    while (isl_union_set_is_empty(Producers) == isl_bool_false)
    {
        for (auto c : this->get_computations())
        {
            DEBUG(3, tiramisu::str_dump("Computing the domain (bounds) of the computation: " + c->get_name()));
            isl_union_set *c_dom = isl_union_set_from_set(isl_set_copy(c->get_iteration_domain()));
            DEBUG(3, tiramisu::str_dump("Domain of the computation: ", isl_union_set_to_str(c_dom)));
            isl_union_set *prods = isl_union_set_copy(Producers);
            DEBUG(3, tiramisu::str_dump("Producers : ", isl_union_set_to_str(prods)));
            // Filter the producers to remove the domains of all the computations except the domain of s1
            // Keep only the computations that have the same space as s1.
            isl_union_set *filter = isl_union_set_universe(isl_union_set_copy(c_dom));
            isl_union_set *c_prods = isl_union_set_intersect(isl_union_set_copy(filter), prods);
            DEBUG(3, tiramisu::str_dump("After keeping only the producers that have the same space as the domain.",
                                        isl_union_set_to_str(c_prods)));

            // If this is not an update operation, we can update its domain, otherwise
            // we do not update the domain and keep the one provided by the user.
            if (c->has_multiple_definitions() == false)
            {
                // REC TODO: in the documentation of compute_bounds indicate that compute_bounds does not update the bounds of update operations
                if ((isl_union_set_is_empty(c_prods) == isl_bool_false))
                {
                    if ((isl_set_plain_is_universe(c->get_iteration_domain()) == isl_bool_true))
                    {
                        DEBUG(3, tiramisu::str_dump("The iteration domain of the computation is a universe."));
                        DEBUG(3, tiramisu::str_dump("The new domain of the computation = ",
                                                    isl_union_set_to_str(c_prods)));
                        c->set_iteration_domain(isl_set_from_union_set(isl_union_set_copy(c_prods)));
                    }
                    else
                    {
                        DEBUG(3, tiramisu::str_dump("The iteration domain of the computation is NOT a universe."));
                        isl_union_set *u = isl_union_set_union(isl_union_set_copy(c_prods),
                                                               isl_union_set_copy(c_dom));
                        c->set_iteration_domain(isl_set_from_union_set(isl_union_set_copy(u)));
                        DEBUG(3, tiramisu::str_dump("The new domain of the computation = ",
                                                    isl_union_set_to_str(u)));
                    }
                }
            }
            else
            {
                assert((isl_set_plain_is_universe(c->get_iteration_domain()) == isl_bool_false) &&
                       "The iteration domain of an update should not be universe.");
                assert((isl_set_is_empty(c->get_iteration_domain()) == isl_bool_false) &&
                       "The iteration domain of an update should not be empty.");
            }

            DEBUG(3, tiramisu::str_dump(""));
        }

        old_Producers = isl_union_set_copy(Producers);
        Producers = isl_union_set_apply(isl_union_set_copy(Producers), isl_union_map_copy(Reverse));
        DEBUG(3, tiramisu::str_dump("The new Producers : ", isl_union_set_to_str(Producers)));

        // If the graph of dependences has recursive dependences, then the intersection of
        // the old producers and the new producers will not be empty (i.e., the old producer and the new producer
        // are the same).
        // In this case, we should subtract the common domain so that in the next iterations of the
        // the algorithm we do not get the same computation again and again (since we have a recursive dependence).
        // This is equivalent to removing the recursive dependence (we remove its image instead of removing it).
        intersection = isl_union_set_intersect(old_Producers, isl_union_set_copy(Producers));
        if (isl_union_set_is_empty(intersection) == isl_bool_false)
        {
            isl_union_set *common_computations = isl_union_set_universe(intersection);
            Producers = isl_union_set_subtract(Producers, common_computations);
            DEBUG(3, tiramisu::str_dump("After eliminating the effect of recursive dependences.",
                                        isl_union_set_to_str(Producers)));
        }

    }
    DEBUG_INDENT(-4);

    DEBUG(3, tiramisu::str_dump("After propagating bounds:"));
    for (auto c : this->get_computations())
    {
        DEBUG(3, tiramisu::str_dump(isl_set_to_str(c->get_iteration_domain())));
    }

    DEBUG_INDENT(-4);
    DEBUG(3, tiramisu::str_dump("End of function"));
}

tiramisu::computation *computation::get_root_of_definition_tree()
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    DEBUG(10, tiramisu::str_dump("Getting the root of the definition tree of this computation: " + this->get_name()));
    DEBUG(10, tiramisu::str_dump("This computation has an ID = " + std::to_string(this->definition_ID)));

    tiramisu::computation *root = this;

    // We know that any child definition has an ID > 0 (since only the root has
    // an ID == 0). So we keep traversing the tree up from the leaf to the root
    // until we find the root. The root is identified by ID == 0.
    while (root->definition_ID > 0)
    {
	root = root->get_first_definition();
        DEBUG(10, tiramisu::str_dump("This computation is: " + root->get_name()));
        DEBUG(10, tiramisu::str_dump("This computation has an ID = " + std::to_string(root->definition_ID)));
    }

    DEBUG(10, tiramisu::str_dump("The root of the tree of updates is: " + root->get_name()));

    DEBUG_INDENT(-4);

    return root;
}

void tiramisu::computation::add_definitions(std::string iteration_domain_str,
        tiramisu::expr e,
        bool schedule_this_computation, tiramisu::primitive_t t,
        tiramisu::function *fct)
{
    tiramisu::computation *new_c = new tiramisu::computation(iteration_domain_str, e,
                                                      schedule_this_computation, t, fct);
    new_c->is_first = false;
    new_c->first_definition = this;
    new_c->is_let = this->is_let;
    new_c->definition_ID = this->definitions_number;
    this->definitions_number++;

    if (new_c->get_expr().is_equal(this->get_expr()))
    {
    	// Copy the associated let statements to the new definition.
    	new_c->associated_let_stmts = this->associated_let_stmts;
    }

    this->updates.push_back(new_c);
}

void tiramisu::function::dump_dep_graph()
{

    tiramisu::str_dump("Dependence graph:\n");
    isl_union_map *deps = isl_union_map_copy(this->compute_dep_graph());
    isl_union_map_dump(deps);
}

/**
  * Return a map that represents the buffers of the function.
  * The buffers of the function are buffers that are either passed
  * to the function as arguments or are buffers that are declared
  * and allocated within the function itself.
  * The names of the buffers are used as a key for the map.
  */
// @{
const std::map<std::string, tiramisu::buffer *> &function::get_buffers() const
{
    return buffers_list;
}
// @}

/**
   * Return a vector of the computations of the function.
   * The order of the computations in the vector does not have any
   * effect on the actual order of execution of the computations.
   * The order of execution of computations is specified through the
   * schedule.
   */
// @{
const std::vector<computation *> &function::get_computations() const
{
    return body;
}
// @}

/**
  * Return the context of the function. i.e. an ISL set that
  * represents constraints over the parameters of the functions
  * (a parameter is an invariant of the function).
  * An example of a context set is the following:
  *          "[N,M]->{: M>0 and N>0}"
  * This context set indicates that the two parameters N and M
  * are strictly positive.
  */
isl_set *function::get_program_context() const
{
    if (context_set != NULL)
    {
        return isl_set_copy(context_set);
    }
    else
    {
        return NULL;
    }
}

/**
  * Get the name of the function.
  */
const std::string &function::get_name() const
{
    return name;
}

/**
  * Return a vector representing the invariants of the function
  * (symbolic constants or variables that are invariant to the
  * function i.e. do not change their value during the execution
  * of the function).
  */
// @{
const std::vector<tiramisu::constant> &function::get_invariants() const
{
    return invariants;
}
// @}

/**
  * Return the Halide statement that represents the whole
  * function.
  * The Halide statement is generated by the code generator.
  * This function should not be called before calling the code
  * generator.
  */
Halide::Internal::Stmt function::get_halide_stmt() const
{
    assert(halide_stmt.defined() && ("Empty Halide statement"));

    return halide_stmt;
}

/**
  * Return the isl context associated with this function.
  */
isl_ctx *function::get_isl_ctx() const
{
    return ctx;
}

/**
  * Return the isl ast associated with this function.
  */
isl_ast_node *function::get_isl_ast() const
{
    assert((ast != NULL) && ("You should generate an ISL ast first (gen_isl_ast())."));

    return ast;
}

/**
  * Get the iterator names of the function.
  */
const std::vector<std::string> &function::get_iterator_names() const
{
    return iterator_names;
}

/**
  * Return true if the computation \p comp should be parallelized
  * at the loop level \p lev.
  */
bool function::should_parallelize(const std::string &comp, int lev) const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev >= 0);

    bool found = false;

    DEBUG(10, tiramisu::str_dump("Checking if the computation " + comp +
                                 " should be parallelized" +
                                 " at the loop level " + std::to_string(lev)));

    for (const auto &pd : this->parallel_dimensions)
    {
        DEBUG(10, tiramisu::str_dump("Checking if the computation " + comp +
                                     " at the loop level " + std::to_string(lev) +
                                     " is equal to the tagged computation " +
                                     pd.first + " at the level " + std::to_string(pd.second)));

        if ((pd.first == comp) && (pd.second == lev))
        {
            found = true;

            DEBUG(10, tiramisu::str_dump("Yes equal."));
        }
    }

    std::string str = "Dimension " + std::to_string(lev) +
                      (found ? " should" : " should not")
                       + " be mapped to CPU thread.";
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);

    return found;
}

/**
* Return the vector length of the computation \p comp at
* at the loop level \p lev.
*/
int function::get_vector_length(const std::string &comp, int lev) const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev >= 0);

    int vector_length = -1;
    bool found = false;

    for (const auto &pd : this->vector_dimensions)
    {
        if ((std::get<0>(pd) == comp) && (std::get<1>(pd) == lev))
        {
	   vector_length = std::get<2>(pd);
	   found = true;
        }
    }

    std::string str = "Dimension " + std::to_string(lev) +
                      (found ? " should" : " should not")
                       + " be vectorized with a vector length of " +
		       std::to_string(vector_length);
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);

    return vector_length;
}


/**
* Return true if the computation \p comp should be vectorized
* at the loop level \p lev.
*/
bool function::should_vectorize(const std::string &comp, int lev) const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev >= 0);

    bool found = false;

    DEBUG(10, tiramisu::str_dump("Checking if the computation " + comp +
                                 " should be vectorized" +
                                 " at the loop level " + std::to_string(lev)));

    DEBUG_INDENT(4);

    for (const auto &pd : this->vector_dimensions)
    {
        DEBUG(10, tiramisu::str_dump("Comparing " + comp + " to " + std::get<0>(pd)));
        DEBUG(10, tiramisu::str_dump(std::get<0>(pd) + " is marked for vectorization at level " + std::to_string(std::get<1>(pd))));

        if ((std::get<0>(pd) == comp) && (std::get<1>(pd) == lev))
            found = true;
    }

    std::string str = "Dimension " + std::to_string(lev) +
                      (found ? " should" : " should not")
                       + " be vectorized.";
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);
    DEBUG_INDENT(-4);

    return found;
}

bool function::should_distribute(const std::string &comp, int lev) const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev >= 0);

    bool found = false;

    DEBUG(10, tiramisu::str_dump("Checking if the computation " + comp +
                                 " should be distributed" +
                                 " at the loop level " + std::to_string(lev)));

    for (const auto &pd : this->distributed_dimensions)
    {
        DEBUG(10, tiramisu::str_dump("Comparing " + comp + " to " + std::get<0>(pd)));
        DEBUG(10, tiramisu::str_dump(std::get<0>(pd) + " is marked for distribution at level " + std::to_string(std::get<1>(pd))));

        if ((std::get<0>(pd) == comp) && (std::get<1>(pd) == lev))
            found = true;
    }

    std::string str = "Dimension " + std::to_string(lev) +
                      (found ? " should" : " should not")
                      + " be distributed.";
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);

    return found;
}

bool tiramisu::function::needs_rank_call() const
{
    return _needs_rank_call;
}

void function::set_context_set(isl_set *context)
{
    assert((context != NULL) && "Context is NULL");

    this->context_set = context;
}

void function::set_context_set(const std::string &context_str)
{
    assert((!context_str.empty()) && "Context string is empty");

    this->context_set = isl_set_read_from_str(this->get_isl_ctx(), context_str.c_str());
    assert((context_set != NULL) && "Context set is NULL");
}

void function::add_context_constraints(const std::string &context_str)
{
    assert((!context_str.empty()) && "Context string is empty");

    if (this->context_set != NULL)
    {
        this->context_set =
            isl_set_intersect(this->context_set,
                              isl_set_read_from_str(this->get_isl_ctx(), context_str.c_str()));
    }
    else
    {
        this->context_set = isl_set_read_from_str(this->get_isl_ctx(), context_str.c_str());
    }
    assert((context_set != NULL) && "Context set is NULL");
}

/**
  * Set the iterator names of the function.
  */
void function::set_iterator_names(const std::vector<std::string> &iteratorNames)
{
    iterator_names = iteratorNames;
}

/**
  * Add an iterator to the function.
  */
void function::add_iterator_name(const std::string &iteratorName)
{
    iterator_names.push_back(iteratorName);
}

/**
  * Generate an object file.  This object file will contain the compiled
  * function.
  * \p obj_file_name indicates the name of the generated file.
  * \p os indicates the target operating system.
  * \p arch indicates the architecture of the target (the instruction set).
  * \p bits indicate the bit-width of the target machine.
  *    must be 0 for unknown, or 32 or 64.
  * For a full list of supported value for \p os and \p arch please
  * check the documentation of Halide::Target
  * (http://halide-lang.org/docs/struct_halide_1_1_target.html).
  * If the machine parameters are not supplied, it will detect one automatically.
  */
// @{
void function::gen_halide_obj(const std::string &obj_file_name) const
{
    Halide::Target target = Halide::get_host_target();
    gen_halide_obj(obj_file_name, target.os, target.arch, target.bits);
}
// @}


void tiramisu::computation::rename_computation(std::string new_name)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(this->get_function()->get_computation_by_name(new_name).empty());

    std::string old_name = this->get_name();

    this->set_name(new_name);

    // Rename the iteration domain.
    isl_set *dom = this->get_iteration_domain();
    assert(dom != NULL);
    dom = isl_set_set_tuple_name(dom, new_name.c_str());
    DEBUG(10, tiramisu::str_dump("Setting the iteration domain to ", isl_set_to_str(dom)));
    this->set_iteration_domain(dom);

    // Rename the time-space domain (if it is not NULL)
    dom = this->get_time_processor_domain();
    if (dom != NULL)
    {
        dom = isl_set_set_tuple_name(dom, new_name.c_str());
        DEBUG(10, tiramisu::str_dump("Setting the time-space domain to ", isl_set_to_str(dom)));
        this->time_processor_domain = dom;
    }

    if (this->get_access_relation() != NULL)
    {
	// Rename the access relation of the computation.
	isl_map *access = this->get_access_relation();
	access = isl_map_set_tuple_name(access, isl_dim_in, new_name.c_str());
	DEBUG(10, tiramisu::str_dump("Setting the access relation to ", isl_map_to_str(access)));
	this->set_access(access);
    }

    // Rename the schedule
    isl_map *sched = this->get_schedule();
    sched = isl_map_set_tuple_name(sched, isl_dim_in, new_name.c_str());
    sched = isl_map_set_tuple_name(sched, isl_dim_out, new_name.c_str());
    DEBUG(10, tiramisu::str_dump("Setting the schedule relation to ", isl_map_to_str(sched)));
    this->set_schedule(sched);

    // Rename parallel, unroll, vectorize and gpu vectors
    for (auto &pd : this->get_function()->unroll_dimensions)
        if (pd.first == old_name)
            pd.first = new_name;
    for (auto &pd : this->get_function()->parallel_dimensions)
        if (pd.first == old_name)
            pd.first = new_name;
    for (auto &pd : this->get_function()->gpu_block_dimensions)
        if (pd.first == old_name)
            pd.first = new_name;
    for (auto &pd : this->get_function()->gpu_thread_dimensions)
        if (pd.first == old_name)
            pd.first = new_name;
    for (auto &pd : this->get_function()->vector_dimensions)
        if (std::get<0>(pd) == old_name)
            std::get<0>(pd) = new_name;

    DEBUG_INDENT(-4);
}


/**
 * A pass to rename computations.
 * Computation that are defined multiple times need to be renamed, because
 * those computations in general have different expressions and the code
 * generator expects that computations that have the same name always have
 * the same expression and access relation. So we should rename them to avoid
 * any ambiguity for the code generator.
 *
 */
void tiramisu::function::rename_computations()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Computations that have been defined multiple times should
    // be renamed. ISL code generator expects computations with the
    // same name to have the same expressions and the same access
    // relation. So, "update" computations that have the same name
    // but have different expressions should be renamed first so
    // that we can use the original code generator without any
    // modification.
    for (auto const comp : this->get_computations())
    {
        std::vector<tiramisu::computation *> same_name_computations =
            this->get_computation_by_name(comp->get_name());

        int i = 0;

        if (same_name_computations.size() > 1)
            for (auto c : same_name_computations)
            {
                std::string new_name = "_" + c->get_name() + "_update_" + std::to_string(i);
                c->rename_computation(new_name);
                i++;
            }
    }

    DEBUG(3, tiramisu::str_dump("After renaming the computations."));
    DEBUG(3, this->dump(false));

    DEBUG_INDENT(-4);
}

/**
  * Generate an isl AST for the function.
  */
void function::gen_isl_ast()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Check that time_processor representation has already been computed,
    assert(this->get_trimmed_time_processor_domain() != NULL);
    assert(this->get_aligned_identity_schedules() != NULL);

    isl_ctx *ctx = this->get_isl_ctx();
    assert(ctx != NULL);
    isl_ast_build *ast_build;

    // Rename updates so that they have different names because
    // the code generator expects each unique name to have
    // an expression, different computations that have the same
    // name cannot have different expressions.
    this->rename_computations();

    if (this->get_program_context() == NULL)
    {
        ast_build = isl_ast_build_alloc(ctx);
    }
    else
    {
        ast_build = isl_ast_build_from_context(isl_set_copy(this->get_program_context()));
    }

    isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
    isl_options_get_ast_build_exploit_nested_bounds(ctx);
    isl_options_set_ast_build_group_coscheduled(ctx, 1);

    ast_build = isl_ast_build_set_after_each_for(ast_build, &tiramisu::for_code_generator_after_for,
                NULL);
    ast_build = isl_ast_build_set_at_each_domain(ast_build, &tiramisu::generator::stmt_code_generator,
                this);

    // Set iterator names
    isl_id_list *iterators = isl_id_list_alloc(ctx, this->get_iterator_names().size());
    if (this->get_iterator_names().size() > 0)
    {
        std::string name = generate_new_variable_name();
        isl_id *id = isl_id_alloc(ctx, name.c_str(), NULL);
        iterators = isl_id_list_add(iterators, id);

        for (int i = 0; i < this->get_iterator_names().size(); i++)
        {
            name = this->get_iterator_names()[i];
            id = isl_id_alloc(ctx, name.c_str(), NULL);
            iterators = isl_id_list_add(iterators, id);

            name = generate_new_variable_name();
            id = isl_id_alloc(ctx, name.c_str(), NULL);
            iterators = isl_id_list_add(iterators, id);
        }

        ast_build = isl_ast_build_set_iterators(ast_build, iterators);
    }

    // Intersect the iteration domain with the domain of the schedule.
    isl_union_map *umap =
        isl_union_map_intersect_domain(
            isl_union_map_copy(this->get_aligned_identity_schedules()),
            isl_union_set_copy(this->get_trimmed_time_processor_domain()));

    DEBUG(3, tiramisu::str_dump("Schedule:", isl_union_map_to_str(this->get_schedule())));
    DEBUG(3, tiramisu::str_dump("Iteration domain:",
                                isl_union_set_to_str(this->get_iteration_domain())));
    DEBUG(3, tiramisu::str_dump("Trimmed Time-Processor domain:",
                                isl_union_set_to_str(this->get_trimmed_time_processor_domain())));
    DEBUG(3, tiramisu::str_dump("Trimmed Time-Processor aligned identity schedule:",
                                isl_union_map_to_str(this->get_aligned_identity_schedules())));
    DEBUG(3, tiramisu::str_dump("Identity schedule intersect trimmed Time-Processor domain:",
                                isl_union_map_to_str(umap)));
    DEBUG(3, tiramisu::str_dump("\n"));

    {
        const char name[] = "atomic";
        isl_space *space;
        isl_union_set *domain, *target;
        isl_union_map *option;

        space = isl_space_set_alloc(ctx, 0, 1);
        space = isl_space_set_tuple_name(space, isl_dim_set, name);
        target = isl_union_set_from_set(isl_set_universe(space));

        domain = isl_union_map_range(isl_union_map_copy(umap));
        domain = isl_union_set_universe(domain);
        option = isl_union_map_from_domain_and_range(domain, target);
        std::cout << isl_union_map_to_str(option) << std::endl;
        ast_build = isl_ast_build_set_options(ast_build, option);
    }

    this->ast = isl_ast_build_node_from_schedule_map(ast_build, umap);

    isl_ast_build_free(ast_build);

    DEBUG_INDENT(-4);
}

std::string generate_new_variable_name()
{
    return "t" + std::to_string(id_counter++);
}

void computation::tag_gpu_level(tiramisu::var L0_var, tiramisu::var L1_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];

    this->tag_gpu_level(L0, L1);

    DEBUG_INDENT(-4);
}

void computation::tag_gpu_level(tiramisu::var L0_var, tiramisu::var L1_var,
	tiramisu::var L2_var, tiramisu::var L3_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(L2_var.get_name().length() > 0);
    assert(L3_var.get_name().length() > 0);

    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name(),
							   L2_var.get_name(), L3_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];
    int L2 = dimensions[2];
    int L3 = dimensions[3];

    this->tag_gpu_level(L0, L1, L2, L3);

    DEBUG_INDENT(-4);
}

void computation::tag_gpu_level(tiramisu::var L0_var, tiramisu::var L1_var,
	tiramisu::var L2_var, tiramisu::var L3_var,
	tiramisu::var L4_var, tiramisu::var L5_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(L2_var.get_name().length() > 0);
    assert(L3_var.get_name().length() > 0);
    assert(L4_var.get_name().length() > 0);
    assert(L5_var.get_name().length() > 0);

    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name(),
							   L2_var.get_name(), L3_var.get_name(),
							   L4_var.get_name(), L5_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];
    int L2 = dimensions[2];
    int L3 = dimensions[3];
    int L4 = dimensions[4];
    int L5 = dimensions[5];

    this->tag_gpu_level(L0, L1, L2, L3, L4, L5);

    DEBUG_INDENT(-4);
}

/**
  * Methods for the computation class.
  */
void tiramisu::computation::parallelize(tiramisu::var par_dim_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(par_dim_var.get_name().length() > 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({par_dim_var.get_name()});
    this->check_dimensions_validity(dimensions);

    int par_dim = dimensions[0];
    this->tag_parallel_level(par_dim);

    DEBUG_INDENT(-4);
}


void tiramisu::computation::tag_parallel_level(int par_dim)
{
    assert(par_dim >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    this->get_function()->add_parallel_dimension(this->get_name(), par_dim);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_gpu_level(int dim0, int dim1)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_block_dimensions(this->get_name(), dim0, -1, -1);
    this->get_function()->add_gpu_thread_dimensions(this->get_name(), dim1, -1, -1);
}

void tiramisu::computation::tag_gpu_level(int dim0, int dim1, int dim2, int dim3)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_block_dimensions(this->get_name(), dim0, dim1, -1);
    this->get_function()->add_gpu_thread_dimensions(this->get_name(), dim2, dim3, -1);
}

void tiramisu::computation::tag_gpu_level(int dim0, int dim1, int dim2, int dim3, int dim4,
        int dim5)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_block_dimensions(this->get_name(), dim0, dim1, dim2);
    this->get_function()->add_gpu_thread_dimensions(this->get_name(), dim3, dim4, dim5);
}

void tiramisu::computation::tag_gpu_block_level(int dim0)
{
    assert(dim0 >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_block_dimensions(this->get_name(), dim0, -1, -1);
}

void tiramisu::computation::tag_gpu_block_level(int dim0, int dim1)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_block_dimensions(this->get_name(), dim0, dim1, -1);
}

void tiramisu::computation::tag_gpu_block_level(int dim0, int dim1, int dim2)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_block_dimensions(this->get_name(), dim0, dim1, dim2);
}

void tiramisu::computation::tag_gpu_thread_level(int dim0)
{
    assert(dim0 >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_thread_dimensions(this->get_name(), dim0, -1);
}

void tiramisu::computation::tag_gpu_thread_level(int dim0, int dim1)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_thread_dimensions(this->get_name(), dim0, dim1);
}

void tiramisu::computation::tag_gpu_thread_level(int dim0, int dim1, int dim2)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_thread_dimensions(this->get_name(), dim0, dim1, dim2);
}

void tiramisu::computation::tag_vector_level(int dim, int length)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(dim >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);
    assert(length > 0);

    this->get_function()->add_vector_dimension(this->get_name(), dim, length);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_vector_level(tiramisu::var L0_var, int v)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    this->tag_vector_level(L0, v);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_distribute_level(tiramisu::var L)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L.get_name().length() > 0);
    std::vector<int> dimensions =
            this->get_loop_level_numbers_from_dimension_names({L.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    this->tag_distribute_level(L0);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_distribute_level(int L)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_distributed_dimension(this->get_name(), L);
    this->get_function()->_needs_rank_call = true;

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_parallel_level(tiramisu::var L0_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    this->tag_parallel_level(L0);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_unroll_level(tiramisu::var L0_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    this->tag_unroll_level(L0);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_unroll_level(int level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(level >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_unroll_dimension(this->get_name(), level);

    DEBUG_INDENT(-4);
}

tiramisu::computation *tiramisu::computation::copy()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::computation *new_c =
        new tiramisu::computation(isl_set_to_str(isl_set_copy(this->get_iteration_domain())),
                                  this->get_expr(),
                                  this->should_schedule_this_computation(),
                                  this->get_data_type(),
                                  this->get_function());

    new_c->set_schedule(isl_map_copy(this->get_schedule()));

    new_c->access = isl_map_copy(this->access);
    new_c->is_let = this->is_let;

    DEBUG_INDENT(-4);

    return new_c;
}

isl_map *isl_map_set_const_dim(isl_map *map, int dim_pos, int val);

std::string computation::get_dimension_name_for_loop_level(int loop_level)
{
	int dim = loop_level_into_dynamic_dimension(loop_level);
	std::string name = isl_map_get_dim_name(this->get_schedule(), isl_dim_out, dim);
	assert(name.size() > 0);
	return name;
}

/*
 * Separate creates a new computation that has exactly the same name
 * and the same iteration domain but the two computations would have
 * different schedules.
 * The schedule of the original computation would restrict it to the
 * domain where the computation is full. The schedule of the separated
 * (new) computation would restrict it to the partial domain (i.e.,
 * the remaining part).
 *
 * Example, if we have a computation
 * {S0[i]: 0<=i<N}
 *
 * The schedule of the original (full) computation would be
 * {S0[i]->S0[0, 0, i, 0]: 0<=i<v*floor(N/v)}
 *
 * The schedule of the separated (partial) computation would be
 * {S0[i]->S0[0, 10, i, 0]: v*floor(N/v)<=i<N}
 *
 * Design choices:
 * - We cannot actually change the iteration domain because this will not compose
 * with the other trasnformations that are expressed are schedules. So we have
 * to actually express the separation transformation using the schedule.
 * - At the same time, we want to be able to manipulate the separated computation
 * and schedule it, so we want to access it with get_update(ID), to make that simple
 * we create a new computation. That is better than just keeping the same original
 * computation and addin a new schedule to it for the separated computation.
 */
void tiramisu::computation::separate(int dim, tiramisu::expr N, int v)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Separating the computation at level " + std::to_string(dim)));

    DEBUG(3, tiramisu::str_dump("Generating the time-space domain."));
    this->gen_time_space_domain();


    //////////////////////////////////////////////////////////////////////////////

    // We create the constraint (i < v*floor(N/v))
    DEBUG(3, tiramisu::str_dump("Constructing the constraint (i<v*floor(N/v))"));
    DEBUG(3, tiramisu::str_dump("Removing any cast operator in N."));
    std::string N_without_cast = N.to_str();
    while (N_without_cast.find("cast") != std::string::npos) // while there is a "cast" in the expression
    {
	    // Remove "cast" from the string, we do not need it.
	    // An alternative to this would be to actually mutate the expression N and remove the cast
	    // operator, but that is more time consuming to implement than replacing the string directly.
	    int pos = N_without_cast.find("cast");
	    N_without_cast = N_without_cast.erase(pos, 4);
    }
 
    std::string constraint;
    constraint = "";
    for (int i=0; i<isl_map_dim(this->get_schedule(), isl_dim_param); i++)
    {
        if (i==0)
            constraint += "[";
	constraint += isl_map_get_dim_name(this->get_schedule(), isl_dim_param, i);
        if (i!=isl_map_dim(this->get_schedule(), isl_dim_param)-1)
            constraint += ",";
        else
            constraint += "]->";
    }
    constraint += "{" + this->get_name() + "[0,";
    for (int i=1; i<isl_map_dim(this->get_schedule(), isl_dim_out); i++)
    {
        if ((i%2==0) && (isl_map_has_dim_name(this->get_schedule(), isl_dim_out, i)==true))
	    constraint += isl_map_get_dim_name(this->get_schedule(), isl_dim_out, i);
        else
	    constraint += "o" + std::to_string(i);
        if (i != isl_map_dim(this->get_schedule(), isl_dim_out)-1)
            constraint += ",";
    }
    constraint += "]: ";

    std::string constraint1 = constraint +
				this->get_dimension_name_for_loop_level(dim) + " < (" + std::to_string(v) + "*(floor((" + N_without_cast + ")/" + std::to_string(v) + ")))}";
    DEBUG(3, tiramisu::str_dump("The constraint is:" + constraint1));

    // We create the constraint (i >= v*floor(N/v))
    DEBUG(3, tiramisu::str_dump("Constructing the constraint (i>=v*(floor(N/v)))"));
    std::string constraint2 = constraint +
				this->get_dimension_name_for_loop_level(dim) + " >= (" + std::to_string(v) + "*(floor((" + N_without_cast + ")/" + std::to_string(v) + ")))}";
    DEBUG(3, tiramisu::str_dump("The constraint is:" + constraint2));

    //////////////////////////////////////////////////////////////////////////////

    isl_set *constraint2_isl = isl_set_read_from_str(this->get_ctx(), constraint2.c_str());

    if (isl_set_is_empty(isl_map_range(isl_map_intersect_range(isl_map_copy(this->get_schedule()), constraint2_isl))) == false)
    {
	    DEBUG(3, tiramisu::str_dump("The separate computation is not empty."));

	    // Create the separated computation.
	    // First, create the domain of the separated computation (which is identical to
	    // the domain of the original computation). Both also have the same name.
	    // TODO: create copy functions for all the classes so that we can copy the objects
	    // we need to have this->get_expr().copy()

	    std::string domain_str = std::string(isl_set_to_str(this->get_iteration_domain()));
	    this->add_definitions(domain_str,
		    this->get_expr(),
		    this->should_schedule_this_computation(),
		    this->get_data_type(),
		    this->get_function());

	    // Set the schedule of the newly created computation (separated
	    // computation) to be equal to the schedule of the original computation.
	    isl_map *new_schedule = isl_map_copy(this->get_schedule());
	    this->get_last_update().set_schedule(new_schedule);

	    // Create the access relation of the separated computation.
	    if (this->get_access_relation() != NULL)
	    {
		    DEBUG(3, tiramisu::str_dump("Creating the access function of the separated computation.\n"));
		    this->get_last_update().set_access(isl_map_copy(this->get_access_relation()));

		    DEBUG(3, tiramisu::str_dump("Access of the separated computation:",
						isl_map_to_str(this->get_last_update().get_access_relation())));
	    }

	    this->get_last_update().add_schedule_constraint("", constraint2.c_str());

	    // Mark the separated computation to be executed after the original (full)
	    // computation.
	    this->get_last_update().after(*this, dim);

	    DEBUG(3, tiramisu::str_dump("The separate computation:"); this->get_last_update().dump());
    }
    else
    {
	    DEBUG(3, tiramisu::str_dump("The separate computation is empty. Thus not added."));
    }

    this->add_schedule_constraint("", constraint1.c_str());

    DEBUG(3, tiramisu::str_dump("The original computation:"); this->dump());

    DEBUG_INDENT(-4);
}

void tiramisu::computation::separate_at(var _level, std::vector<tiramisu::expr> _separate_points, tiramisu::expr _max)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Separating the computation at level " + _level.get_name()));

    DEBUG(3, tiramisu::str_dump("Generating the time-space domain."));
    this->gen_time_space_domain();

    std::vector<int> dimensions =
            this->get_loop_level_numbers_from_dimension_names({_level.get_name()});
    int level = dimensions[0];

    //////////////////////////////////////////////////////////////////////////////

    std::vector<tiramisu::constant> separate_points;
    for (auto p : _separate_points) {
        separate_points.push_back(tiramisu::constant("c" + std::to_string(id_counter++), p, p.get_data_type(), true,
                                                     NULL, 0, this->get_function()));
    }
    tiramisu::constant max("c" + std::to_string(id_counter++), _max, _max.get_data_type(), true, NULL, 0,
                           this->get_function());

    // We create the constraint (i < separate_point)
    DEBUG(3, tiramisu::str_dump("Constructing the constraint (i<middle)"));
    std::string constraint;
    constraint = "";
    // get the constants
    for (int i=0; i<isl_map_dim(this->get_schedule(), isl_dim_param); i++)
    {
        if (i==0) {
            constraint += "[" + max.get_name() + ",";
            for (auto separate_point : separate_points) {
                constraint += separate_point.get_name() + ",";
            }
        }
        constraint += isl_map_get_dim_name(this->get_schedule(), isl_dim_param, i);
        if (i!=isl_map_dim(this->get_schedule(), isl_dim_param)-1)
            constraint += ",";
        else
            constraint += "]->";
    }
    if (isl_map_dim(this->get_schedule(), isl_dim_param) == 0) {
        // Need to add in these constants
        constraint += "[" + max.get_name();
        for (auto separate_point : separate_points) {
            constraint += ", " +  separate_point.get_name() ;
        }
        constraint += "]->";
    }
    constraint += "{" + this->get_name() + "[0,";
    for (int i=1; i<isl_map_dim(this->get_schedule(), isl_dim_out); i++)
    {
        if ((i%2==0) && (isl_map_has_dim_name(this->get_schedule(), isl_dim_out, i)==true))
            constraint += isl_map_get_dim_name(this->get_schedule(), isl_dim_out, i);
        else
            constraint += "o" + std::to_string(i);
        if (i != isl_map_dim(this->get_schedule(), isl_dim_out)-1)
            constraint += ",";
    }
    constraint += "]: ";
    std::vector<std::string> constraints;
    // This is the first constraint
    std::string constraint1 = constraint +
                              this->get_dimension_name_for_loop_level(level) + " < " + separate_points[0].get_name() + "}";
    DEBUG(3, tiramisu::str_dump("The constraint is:" + constraint1));

    // We create the constraint (i >= separate_point). This is the last constraint
    DEBUG(3, tiramisu::str_dump("Constructing the constraint (i>=middle)"));
    std::string constraintn = constraint +
                              this->get_dimension_name_for_loop_level(level) + " >= " +
                              separate_points[separate_points.size() - 1].get_name() + " and " +
                              this->get_dimension_name_for_loop_level(level) + " < " + max.get_name() + "}";
    DEBUG(3, tiramisu::str_dump("The constraint is:" + constraintn));


    // create the intermediate constraints
    for (int i = 1; i < separate_points.size(); i++) {
        std::string cons = constraint +
                           this->get_dimension_name_for_loop_level(level) + " >= " + separate_points[i-1].get_name() + " and ";
        cons += this->get_dimension_name_for_loop_level(level) + " < " + separate_points[i].get_name() + "}";
        constraints.push_back(cons);
    }
    constraints.push_back(constraintn);
    //////////////////////////////////////////////////////////////////////////////

    for (std::string cons : constraints) {
        isl_set *cons_isl = isl_set_read_from_str(this->get_ctx(), cons.c_str());
        if (isl_set_is_empty(
                isl_map_range(isl_map_intersect_range(isl_map_copy(this->get_schedule()), cons_isl))) == false) {
            DEBUG(3, tiramisu::str_dump("The separate computation is not empty."));

            // Create the separated computation.
            // First, create the domain of the separated computation (which is identical to
            // the domain of the original computation). Both also have the same name.
            // TODO: create copy functions for all the classes so that we can copy the objects
            // we need to have this->get_expr().copy()
            int last_update_computation = this->get_updates().size();

            std::string domain_str = std::string(isl_set_to_str(this->get_iteration_domain()));
            this->add_definitions(domain_str,
                                  this->get_expr(),
                                  this->should_schedule_this_computation(),
                                  this->get_data_type(),
                                  this->get_function());

            // Set the schedule of the newly created computation (separated
            // computation) to be equal to the schedule of the original computation.
            isl_map *new_schedule = isl_map_copy(this->get_schedule());
            this->get_update(last_update_computation).set_schedule(new_schedule);

            // Create the access relation of the separated computation (by replacing its name).
            if (this->get_access_relation() != NULL) {
                DEBUG(3, tiramisu::str_dump("Creating the access function of the separated computation.\n"));
                this->get_update(last_update_computation).set_access(isl_map_copy(this->get_access_relation()));

                DEBUG(3, tiramisu::str_dump("Access of the separated computation:",
                                            isl_map_to_str(
                                                    this->get_update(last_update_computation).get_access_relation())));
            }

            this->get_update(last_update_computation).add_schedule_constraint("", cons.c_str());

            DEBUG(3, tiramisu::str_dump("The separate computation:");
                    this->get_update(last_update_computation).dump());
        } else {
            DEBUG(3, tiramisu::str_dump("The separate computation is empty. Thus not added."));
        }
    }

    this->add_schedule_constraint("", constraint1.c_str());

    DEBUG(3, tiramisu::str_dump("The original computation:"); this->dump());

    // rename all the updates by adding '_<ctr>' to the end of the name
    int ctr = 0;
    for (auto comp : this->get_updates()) {
        comp->rename_computation(comp->get_name() + "_" + std::to_string(ctr++));
    }
    DEBUG_INDENT(-4);
}

void tiramisu::computation::set_iteration_domain(isl_set *domain)
{
    this->iteration_domain = domain;
}

std::string utility::get_parameters_list(isl_set *set)
{
    std::string list = "";

    assert(set != NULL);

    for (int i = 0; i < isl_set_dim(set, isl_dim_param); i++)
    {
        list += isl_set_get_dim_name(set, isl_dim_param, i);
        if ((i != isl_set_dim(set, isl_dim_param) - 1))
        {
            list += ",";
        }
    }

    return list;
}

tiramisu::constant *tiramisu::computation::create_separator(const tiramisu::expr &loop_upper_bound, int v)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    /*
     * Create a new Tiramisu constant M = v*floor(N/v). This is the biggest
     * multiple of w that is still smaller than N.  Add this constant to
     * the list of invariants.
     */
    primitive_t f_type, i_type, u_type;
    if (global::get_loop_iterator_data_type() == p_int32)
    {
        f_type = p_float32;
        i_type = p_int32;
        u_type = p_uint32;
    }
    else
    {
        f_type = p_float64;
        i_type = p_int64;
        u_type = p_uint64;
    }

    std::string separator_name = tiramisu::generate_new_variable_name();
    tiramisu::expr div_expr = tiramisu::expr(o_div, loop_upper_bound, tiramisu::expr(v));
    tiramisu::expr cast_expr = tiramisu::expr(o_cast, f_type, div_expr);
    tiramisu::expr floor_expr = tiramisu::expr(o_floor, cast_expr);
    tiramisu::expr cast2_expr = tiramisu::expr(o_cast, i_type, floor_expr);
    tiramisu::expr separator_expr = tiramisu::expr(o_mul, tiramisu::expr(v), cast2_expr);
    tiramisu::constant *separation_param = new tiramisu::constant(
        separator_name, separator_expr, u_type, true, NULL, 0, this->get_function());

    DEBUG_INDENT(-4);

    return separation_param;
}

tiramisu::buffer *tiramisu::computation::get_automatically_allocated_buffer()
{
    return this->automatically_allocated_buffer;
}

std::vector<tiramisu::expr>* computation::compute_buffer_size()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::vector<tiramisu::expr> *dim_sizes = new std::vector<tiramisu::expr>();

    // If the computation has an update, we first compute the union of all the
    // updates, then we compute the bounds of the union.
    for (int i = 0; i < this->get_iteration_domain_dimensions_number(); i++)
    {
	isl_set *union_iter_domain = isl_set_copy(this->get_update(0).get_iteration_domain());

	for (int j = 1; j < this->get_updates().size(); j++)
	{
            isl_set *iter_domain = isl_set_copy(this->get_update(j).get_iteration_domain());
	    union_iter_domain = isl_set_union(union_iter_domain, iter_domain);
	}

        DEBUG(3, tiramisu::str_dump("Extracting bounds of the following set:", isl_set_to_str(union_iter_domain)));
        tiramisu::expr lower = utility::get_bound(union_iter_domain, i, false);
        tiramisu::expr upper = utility::get_bound(union_iter_domain, i, true);
        tiramisu::expr diff = (upper - lower + 1);

        DEBUG(3, tiramisu::str_dump("Buffer dimension size (dim = " + std::to_string(i) + ") : "); diff.dump(false));
        dim_sizes->push_back(diff);
    }

    DEBUG_INDENT(-4);

    return dim_sizes;
}

/**
 * Algorithm:
 * - Compute the size of the buffer:
 *      - TODO: Future work Use the same code that computes the needed area in compute_at,
 *      - TODO: From the needed area, deduce the size by computing the upper
 *              bound and the lower bound and subtracting the two.
 * - declare a buffer with a random name, and with the computed size,
 * - allocate the buffer and get the computation that allocates the buffer,
 * - map the computation to the allocated buffer (one to one mapping),
 * - schedule the computation that allocates the buffer before \p comp
 * at loop level L0,
 * - return the allocation computation.
 */
tiramisu::computation *computation::store_at(tiramisu::computation &comp,
					    tiramisu::var L0_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    std::vector<tiramisu::expr> *dim_sizes = this->compute_buffer_size();

    tiramisu::buffer *buff = new tiramisu::buffer("_" + this->name + "_buffer",
            (*dim_sizes),
            this->get_data_type(),
            tiramisu::a_temporary,
            this->get_function());

    this->automatically_allocated_buffer = buff;

    tiramisu::computation *allocation = buff->allocate_at(comp, L0);
    this->bind_to(buff);
    if (comp.get_predecessor() != NULL)
	allocation->between(
		*(comp.get_predecessor()),
		L0_var, comp, L0_var);
    else
	allocation->before(comp, L0);

    DEBUG_INDENT(-4);

    return allocation;
}

void tiramisu::computation::vectorize(tiramisu::var L0_var, int v)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::var L0_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L0_inner = tiramisu::var(generate_new_variable_name());
    this->vectorize(L0_var, v, L0_outer, L0_inner);

    DEBUG_INDENT(-4);
}

void computation::update_names(std::vector<std::string> original_loop_level_names, std::vector<std::string> new_names,
			       int erase_from, int nb_loop_levels_to_erase)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("Original loop level names: "));
    for (auto n: original_loop_level_names)
    {
	DEBUG_NO_NEWLINE_NO_INDENT(3, tiramisu::str_dump(n + " "));
    }
    DEBUG_NEWLINE(3);

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("New names: "));
    for (auto n: new_names)
    {
	DEBUG_NO_NEWLINE_NO_INDENT(3, tiramisu::str_dump(n + " "));
    }
    DEBUG_NEWLINE(3);

    DEBUG(3, tiramisu::str_dump("Start erasing from: " + std::to_string(erase_from)));
    DEBUG(3, tiramisu::str_dump("Number of loop levels to erase: " + std::to_string(nb_loop_levels_to_erase)));

    original_loop_level_names.erase(original_loop_level_names.begin() + erase_from, original_loop_level_names.begin() + erase_from + nb_loop_levels_to_erase);

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("Original loop level names after erasing loop levels: "));
    for (auto n: original_loop_level_names)
    {
	DEBUG_NO_NEWLINE_NO_INDENT(3, tiramisu::str_dump(n + " "));
    }
    DEBUG_NEWLINE(3);

    original_loop_level_names.insert(original_loop_level_names.begin() + erase_from, new_names.begin(), new_names.end());

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("Original loop level names after inserting the new loop levels: "));
    for (auto n: original_loop_level_names)
    {
	DEBUG_NO_NEWLINE_NO_INDENT(3, tiramisu::str_dump(n + " "));
    }
    DEBUG_NEWLINE(3);

    this->set_loop_level_names(original_loop_level_names);
//    this->name_unnamed_time_space_dimensions();

    DEBUG(3, tiramisu::str_dump("Names updated. New names are: "));
    for (auto n: this->get_loop_level_names())
    {
	DEBUG_NO_NEWLINE_NO_INDENT(3, tiramisu::str_dump(n + " "));
    }

    DEBUG_INDENT(-4);
}

void tiramisu::computation::vectorize(tiramisu::var L0_var, int v, tiramisu::var L0_outer, tiramisu::var L0_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    bool split_happened = this->separateAndSplit(L0_var, v, L0_outer, L0_inner);

    if (split_happened)
    {
        // Tag the inner loop after splitting to be vectorized. That loop
        // is supposed to have a constant extent.
        this->get_update(0).tag_vector_level(L0 + 1, v);
    }
    else
    {
        this->get_update(0).tag_vector_level(L0, v);
	this->set_loop_level_names({L0}, {L0_outer.get_name()});
    }

    // Replace the original dimension name with two new dimension names
    this->update_names(original_loop_level_names, {L0_outer.get_name(), L0_inner.get_name()}, L0, 1);

    this->get_function()->align_schedules();

    DEBUG_INDENT(-4);
}

tiramisu::computation& computation::get_last_update()
{
	return this->get_update(this->get_updates().size()-1);
}

/**
  * Returns all updates the have been defined for this computation using
  * add_definitions. The 0th update is a pointer to this computation.
  */
std::vector<computation*>& tiramisu::computation::get_updates() {
    return this->updates;
}

/**
  * Returns the \p index update that has been added to this computation such that:
  * - If \p index == 0, then this computation is returned.
  * - If \p > 0, then it returns the pth computation added through add_definitions.
  */
computation& tiramisu::computation::get_update(int i)
{
    return *(this->updates[i]);
}

void tiramisu::computation::unroll(tiramisu::var L0_var, int v)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::var L0_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L0_inner = tiramisu::var(generate_new_variable_name());
    this->unroll(L0_var, v, L0_outer, L0_inner);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::unroll(tiramisu::var L0_var, int v, tiramisu::var L0_outer, tiramisu::var L0_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    bool split_happened = this->separateAndSplit(L0_var, v, L0_outer, L0_inner);

    if (split_happened)
    {
	// Tag the inner loop after splitting to be unrolled. That loop
	// is supposed to have a constant extent.
	this->get_update(0).tag_unroll_level(L0 + 1);
    }
    else
    {
        this->get_update(0).tag_unroll_level(L0);
	this->set_loop_level_names({L0}, {L0_outer.get_name()});
    }

    // Replace the original dimension name with two new dimension names
    this->update_names(original_loop_level_names, {L0_outer.get_name(), L0_inner.get_name()}, L0, 1);

    this->get_function()->align_schedules();

    DEBUG_INDENT(-4);
}

void computation::dump_iteration_domain() const
{
    if (ENABLE_DEBUG)
    {
        isl_set_dump(this->get_iteration_domain());
    }
}

void function::dump_halide_stmt() const
{
    if (ENABLE_DEBUG)
    {
        tiramisu::str_dump("\n\n");
        tiramisu::str_dump("\nGenerated Halide Low Level IR:\n");
        std::cout << this->get_halide_stmt();
        tiramisu::str_dump("\n\n\n\n");
    }
}

void function::dump_trimmed_time_processor_domain() const
{
    // Create time space domain

    if (ENABLE_DEBUG)
    {
        tiramisu::str_dump("\n\nTrimmed Time-processor domain:\n");

        tiramisu::str_dump("Function " + this->get_name() + ":\n");
        for (const auto &comp : this->get_computations())
        {
            isl_set_dump(comp->get_trimmed_time_processor_domain());
        }

        tiramisu::str_dump("\n\n");
    }
}

void function::dump_time_processor_domain() const
{
    // Create time space domain

    if (ENABLE_DEBUG)
    {
        tiramisu::str_dump("\n\nTime-processor domain:\n");

        tiramisu::str_dump("Function " + this->get_name() + ":\n");
        for (const auto &comp : this->get_computations())
        {
            isl_set_dump(comp->get_time_processor_domain());
        }

        tiramisu::str_dump("\n\n");
    }
}

void function::gen_time_space_domain()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Generate the ordering based on calls to .after() and .before().
    this->gen_ordering_schedules();

    this->align_schedules();

    for (auto &comp : this->get_computations())
    {
        comp->gen_time_space_domain();
    }

    DEBUG_INDENT(-4);
}

void computation::dump_schedule() const
{
    DEBUG_INDENT(4);

    if (ENABLE_DEBUG)
    {
        tiramisu::str_dump("Dumping the schedule of the computation " + this->get_name() + " : ");

        std::flush(std::cout);
        isl_map_dump(this->get_schedule());
    }

    DEBUG_INDENT(-4);
}

void computation::dump() const
{
    if (ENABLE_DEBUG)
    {
        std::cout << std::endl << "Dumping the computation \"" + this->get_name() + "\" :" << std::endl;
        std::cout << "Iteration domain of the computation \"" << this->name << "\" : ";
        std::flush(std::cout);
        isl_set_dump(this->get_iteration_domain());
        std::flush(std::cout);
        this->dump_schedule();

        std::flush(std::cout);
        std::cout << "Expression of the computation : "; std::flush(std::cout);
        this->get_expr().dump(false);
        std::cout << std::endl; std::flush(std::cout);

        std::cout << "Access relation of the computation : "; std::flush(std::cout);
        isl_map_dump(this->get_access_relation());
        if (this->get_access_relation() == NULL)
        {
            std::cout << "\n";
        }
        std::flush(std::cout);

        if (this->get_time_processor_domain() != NULL)
        {
            std::cout << "Time-space domain " << std::endl; std::flush(std::cout);
            isl_set_dump(this->get_time_processor_domain());
        }
        else
        {
            std::cout << "Time-space domain : NULL." << std::endl;
        }

        std::cout << "Computation to be scheduled ? " << (this->schedule_this_computation) << std::endl;

        for (const auto &e : this->index_expr)
        {
            tiramisu::str_dump("Access expression:", (const char *)isl_ast_expr_to_C_str(e));
            tiramisu::str_dump("\n");
        }

        tiramisu::str_dump("Halide statement: ");
        if (this->stmt.defined())
        {
            std::cout << this->stmt;
        }
        else
        {
            tiramisu::str_dump("NULL");
        }
        tiramisu::str_dump("\n");
        tiramisu::str_dump("\n");
    }
}


int max_elem(std::vector<int> vec)
{
    int res = -1;

    for (auto v : vec)
    {
        res = std::max(v, res);
    }

    return res;
}

bool buffer::has_constant_extents()
{
    bool constant_extent = true;

    for (size_t i = 0; i < this->get_dim_sizes().size(); i++)
    {
        if (this->get_dim_sizes()[i].get_expr_type() != tiramisu::e_val)
        {
            constant_extent = false;
        }
    }

    return constant_extent;
}

tiramisu::computation *buffer::allocate_at(tiramisu::computation &C, tiramisu::var level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(level.get_name().length() > 0);

    std::vector<int> dimensions =
	C.get_loop_level_numbers_from_dimension_names({level.get_name()});

    assert(dimensions.size() == 1);

    int L0 = dimensions[0];

    C.check_dimensions_validity({L0});

    tiramisu::computation *alloc = this->allocate_at(C, L0);

    DEBUG_INDENT(-4);

    return alloc;
}

tiramisu::computation *buffer::allocate_at(tiramisu::computation &C, int level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(level >= tiramisu::computation::root_dimension);
    assert(level < (int) isl_set_dim(C.get_iteration_domain(), isl_dim_set));

    DEBUG(3, tiramisu::str_dump("Computing the iteration domain for the allocate() operation"));
    DEBUG(3, tiramisu::str_dump("Computation name " + C.get_name() + ", Level = " + std::to_string(level)));

    isl_set *iter = C.get_iteration_domains_of_all_definitions();

    DEBUG(3, tiramisu::str_dump(
              "The union of iteration domains of the computations with which we allocate (all their definitions): ",
              isl_set_to_str(iter)));

    int projection_dimension = level + 1;
    if (projection_dimension != 0)
        iter = isl_set_project_out(isl_set_copy(iter),
                                   isl_dim_set,
                                   projection_dimension,
                                   isl_set_dim(iter, isl_dim_set) - projection_dimension);
    else
    {
        iter = isl_set_read_from_str(C.get_ctx(), "{[0]}");
    }
    std::string new_name = "_allocation_" + C.get_name();
    iter = isl_set_set_tuple_name(iter, new_name.c_str());
    std::string iteration_domain_str = isl_set_to_str(iter);

    DEBUG(3, tiramisu::str_dump(
              "Computed iteration domain for the allocate() operation",
              isl_set_to_str(iter)));

    tiramisu::expr *new_expression = new tiramisu::expr(tiramisu::o_allocate, this->get_name());

    DEBUG(3, tiramisu::str_dump("The expression of the allocation operation"); new_expression->dump(false));

    tiramisu::computation *alloc = new tiramisu::computation(iteration_domain_str,
            *new_expression,
            true, p_none, C.get_function());

    this->set_auto_allocate(false);

    DEBUG(3, tiramisu::str_dump("The computation representing the allocate() operator:");
          alloc->dump());

    DEBUG_INDENT(-4);

    return alloc;
}

void buffer::set_auto_allocate(bool auto_allocation)
{
    this->auto_allocate = auto_allocation;
}

bool buffer::get_auto_allocate()
{
    return this->auto_allocate;
}

void computation::set_schedule(std::string map_str)
{
    assert(!map_str.empty());
    assert(this->ctx != NULL);

    isl_map *map = isl_map_read_from_str(this->ctx, map_str.c_str());
    assert(map != NULL);

    this->set_schedule(map);
}

void computation::apply_transformation_on_schedule(std::string map_str)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(!map_str.empty());
    assert(this->ctx != NULL);

    isl_map *map = isl_map_read_from_str(this->ctx, map_str.c_str());
    assert(map != NULL);

    DEBUG(3, tiramisu::str_dump("Applying the following transformation on the schedule : "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(map)));

    isl_map *sched = this->get_schedule();
    sched = isl_map_apply_range(isl_map_copy(sched), isl_map_copy(map));
    this->set_schedule(sched);

    DEBUG(3, tiramisu::str_dump("Schedule after transformation : "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::apply_transformation_on_schedule_domain(std::string map_str)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(!map_str.empty());
    assert(this->ctx != NULL);

    isl_map *map = isl_map_read_from_str(this->ctx, map_str.c_str());
    assert(map != NULL);

    DEBUG(3, tiramisu::str_dump("Applying the following transformation on the domain of the schedule : "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(map)));

    isl_map *sched = this->get_schedule();
    sched = isl_map_apply_domain(isl_map_copy(sched), isl_map_copy(map));

    this->set_schedule(sched);

    DEBUG(3, tiramisu::str_dump("Schedule after transformation : "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::add_schedule_constraint(std::string domain_constraints,
        std::string range_constraints)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);


    assert(this->ctx != NULL);
    isl_map *sched = this->get_schedule();

    if (!domain_constraints.empty())
    {
        isl_set *domain_cst = isl_set_read_from_str(this->ctx, domain_constraints.c_str());
        assert(domain_cst != NULL);

        DEBUG(3, tiramisu::str_dump("Adding the following constraints to the domain of the schedule : "));
        DEBUG(3, tiramisu::str_dump(isl_set_to_str(domain_cst)));
        DEBUG(3, tiramisu::str_dump("The schedule is : "));
        DEBUG(3, tiramisu::str_dump(isl_map_to_str(sched)));

        sched = isl_map_intersect_domain(isl_map_copy(sched), isl_set_copy(domain_cst));

    }

    if (!range_constraints.empty())
    {
        isl_set *range_cst = isl_set_read_from_str(this->ctx, range_constraints.c_str());

        DEBUG(3, tiramisu::str_dump("Adding the following constraints to the range of the schedule : "));
        DEBUG(3, tiramisu::str_dump(isl_set_to_str(range_cst)));
        DEBUG(3, tiramisu::str_dump("The schedule : ", isl_map_to_str(sched)));

        sched = isl_map_intersect_range(isl_map_copy(sched), isl_set_copy(range_cst));
    }

    this->set_schedule(sched);

    DEBUG(3, tiramisu::str_dump("Schedule after transformation : "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}


/**
  * Set the schedule of the computation.
  *
  * \p map is a string that represents a mapping from the iteration domain
  *  to the time-processor domain (the mapping is in the ISL format:
  *  http://isl.gforge.inria.fr/user.html#Sets-and-Relations).
  *
  * The name of the domain and range space must be identical.
  */
void tiramisu::computation::set_schedule(isl_map *map)
{
    this->schedule = map;
}


void computation::set_low_level_schedule(std::string map_str)
{
    assert(!map_str.empty());
    assert(this->ctx != NULL);

    isl_map *map = isl_map_read_from_str(this->ctx, map_str.c_str());
    assert(map != NULL);

    this->set_low_level_schedule(map);
}

void tiramisu::computation::set_low_level_schedule(isl_map *map)
{
    this->fct->use_low_level_scheduling_commands = true;
    this->set_schedule(map);
}


struct param_pack_1
{
    int in_dim;
    int out_constant;
};

/**
 * Take a basic map as input, go through all of its constraints,
 * identifies the constraint of the static dimension param_pack_1.in_dim
 * (passed in user) and replace the value of param_pack_1.out_constant if
 * the static dimension is bigger than that value.
 */
isl_stat extract_static_dim_value_from_bmap(__isl_take isl_basic_map *bmap, void *user)
{
    struct param_pack_1 *data = (struct param_pack_1 *) user;

    isl_constraint_list *list = isl_basic_map_get_constraint_list(bmap);
    int n_constraints = isl_constraint_list_n_constraint(list);

    for (int i = 0; i < n_constraints; i++)
    {
        isl_constraint *cst = isl_constraint_list_get_constraint(list, i);
        isl_val *val = isl_constraint_get_coefficient_val(cst, isl_dim_out, data->in_dim);
        if (isl_val_is_one(val)) // i.e., the coefficient of the dimension data->in_dim is 1
        {
            isl_val *val2 = isl_constraint_get_constant_val(cst);
            int const_val = (-1) * isl_val_get_num_si(val2);
            data->out_constant = const_val;
            DEBUG(3, tiramisu::str_dump("Dimensions found.  Constant = " +
                                        std::to_string(const_val)));
        }
    }

    return isl_stat_ok;
}

// if multiple const values exist, choose the maximal value among them because we
// want to use this value to know by how much we shift the computations back.
// so we need to figure out the maximal const value and use it to shift the iterations
// backward so that that iteration runs before the consumer.
isl_stat extract_constant_value_from_bset(__isl_take isl_basic_set *bset, void *user)
{
    struct param_pack_1 *data = (struct param_pack_1 *) user;

    isl_constraint_list *list = isl_basic_set_get_constraint_list(bset);
    int n_constraints = isl_constraint_list_n_constraint(list);

    for (int i = 0; i < n_constraints; i++)
    {
        isl_constraint *cst = isl_constraint_list_get_constraint(list, i);
        if (isl_constraint_is_equality(cst) &&
                isl_constraint_involves_dims(cst, isl_dim_set, data->in_dim, 1))
        {
            isl_val *val = isl_constraint_get_coefficient_val(cst, isl_dim_out, data->in_dim);
            assert(isl_val_is_one(val));
            // assert that the coefficients of all the other dimension spaces are zero.

            isl_val *val2 = isl_constraint_get_constant_val(cst);
            int const_val = (-1) * isl_val_get_num_si(val2);
            data->out_constant = std::max(data->out_constant, const_val);
            DEBUG(3, tiramisu::str_dump("Dimensions found.  Constant = " +
                                        std::to_string(const_val)));
        }
    }

    return isl_stat_ok;
}

/**
 * Return the value of the static dimension.
 *
 * For example, if we have a map M = {S0[i,j]->[0,0,i,1,j,2]; S0[i,j]->[1,0,i,1,j,3]}
 * and call isl_map_get_static_dim(M, 5, 1), it will return 3.
 */
int isl_map_get_static_dim(isl_map *map, int dim_pos)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(map != NULL);
    assert(dim_pos >= 0);
    assert(dim_pos <= (signed int) isl_map_dim(map, isl_dim_out));

    DEBUG(3, tiramisu::str_dump("Getting the constant coefficient of ",
                                isl_map_to_str(map));
          tiramisu::str_dump(" at dimension ");
          tiramisu::str_dump(std::to_string(dim_pos)));

    struct param_pack_1 *data = (struct param_pack_1 *) malloc(sizeof(struct param_pack_1));
    data->out_constant = 0;
    data->in_dim = dim_pos;

    isl_map_foreach_basic_map(isl_map_copy(map),
                              &extract_static_dim_value_from_bmap,
                              data);

    DEBUG(3, tiramisu::str_dump("The constant is: ");
          tiramisu::str_dump(std::to_string(data->out_constant)));

    DEBUG_INDENT(-4);

    return data->out_constant;
}

int isl_set_get_const_dim(isl_set *set, int dim_pos)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(set != NULL);
    assert(dim_pos >= 0);
    assert(dim_pos <= (signed int) isl_set_dim(set, isl_dim_out));

    DEBUG(3, tiramisu::str_dump("Getting the constant coefficient of ",
                                isl_set_to_str(set));
          tiramisu::str_dump(" at dimension ");
          tiramisu::str_dump(std::to_string(dim_pos)));

    struct param_pack_1 *data = (struct param_pack_1 *) malloc(sizeof(struct param_pack_1));
    data->out_constant = 0;
    data->in_dim = dim_pos;

    isl_set_foreach_basic_set(isl_set_copy(set),
                              &extract_constant_value_from_bset,
                              data);

    DEBUG(3, tiramisu::str_dump("The constant is: ");
          tiramisu::str_dump(std::to_string(data->out_constant)));

    DEBUG_INDENT(-4);

    return data->out_constant;
}

/**
 * Set the value \p val for the output dimension \p dim_pos of \p map.
 *
 * Example
 *
 * Assuming the map M = {S[i,j]->[i0,i1,i2]}
 *
 * M = isl_map_set_const_dim(M, 0, 0);
 *
 * Would create the constraint i0=0 and add it to the map.
 * The resulting map is
 *
 * M = {S[i,j]->[i0,i1,i2]: i0=0}
 *
 */
isl_map *isl_map_set_const_dim(isl_map *map, int dim_pos, int val)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(map != NULL);
    assert(dim_pos >= 0);
    assert(dim_pos <= (signed int) isl_map_dim(map, isl_dim_out));

    DEBUG(3, tiramisu::str_dump("Setting the constant coefficient of ",
                                isl_map_to_str(map));
          tiramisu::str_dump(" at dimension ");
          tiramisu::str_dump(std::to_string(dim_pos));
          tiramisu::str_dump(" into ");
          tiramisu::str_dump(std::to_string(val)));

    isl_map *identity = isl_set_identity(isl_map_range(isl_map_copy(map)));
    // We need to create a universe of the map (i.e., an unconstrained map)
    // because isl_set_identity() create an identity transformation and
    // inserts the constraints that were in the original set.  We don't
    // want to have those constraints.  We want to have a universe map, i.e.,
    // a map without any constraint.
    identity = isl_map_universe(isl_map_get_space(identity));

    isl_space *sp = isl_map_get_space(identity);
    isl_local_space *lsp = isl_local_space_from_space(isl_space_copy(sp));

    // This loops goes through the output dimensions of the map one by one
    // and adds a constraint for each dimension. IF the dimension is dim_pos
    // it add a constraint of equality to val
    // Otherwise it adds a constraint that keeps the original value, i.e.,
    // (output dimension = input dimension)
    // Example
    //  Assuming that dim_pos = 0, val = 10 and the universe map is
    //  {S[i0,i1]->S[j0,j1]}, this loop produces
    //  {S[i0,i1]->S[j0,j1]: j0=0 and j1=i1}
    //  i.e.,
    //  {S[i0,i1]->S[0,i1]}
    for (int i = 0; i < isl_map_dim(identity, isl_dim_out); i++)
        if (i == dim_pos)
        {
            isl_constraint *cst = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, dim_pos, 1);
            cst = isl_constraint_set_constant_si(cst, (-1) * (val));
            identity = isl_map_add_constraint(identity, cst);
        }
        else
        {
            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
            identity = isl_map_add_constraint(identity, cst2);
        }

    DEBUG(3, tiramisu::str_dump("Transformation map ", isl_map_to_str(identity)));

    map = isl_map_apply_range(map, identity);

    DEBUG(3, tiramisu::str_dump("After applying the transformation map: ",
                                isl_map_to_str(map)));

    DEBUG_INDENT(-4);

    return map;
}

isl_map *isl_map_add_dim_and_eq_constraint(isl_map *map, int dim_pos, int constant)
{
    assert(map != NULL);
    assert(dim_pos >= 0);
    assert(dim_pos <= (signed int) isl_map_dim(map, isl_dim_out));

    map = isl_map_insert_dims(map, isl_dim_out, dim_pos, 1);
    map = isl_map_set_tuple_name(map, isl_dim_out, isl_map_get_tuple_name(map, isl_dim_in));

    isl_space *sp = isl_map_get_space(map);
    isl_local_space *lsp =
        isl_local_space_from_space(isl_space_copy(sp));
    isl_constraint *cst = isl_constraint_alloc_equality(lsp);
    cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, dim_pos, 1);
    cst = isl_constraint_set_constant_si(cst, (-1) * constant);
    map = isl_map_add_constraint(map, cst);

    return map;
}

/**
 * Transform the loop level into its corresponding dynamic schedule
 * dimension.
 *
 * In the example below, the dynamic dimension that corresponds
 * to the loop level 0 is 2, and to 1 it is 4, ...
 *
 * The first dimension is the duplication dimension, the following
 * dimensions are static, dynamic, static, dynamic, ...
 *
 * Loop level               :    -1         0      1      2
 * Schedule dimension number:        0, 1   2  3   4  5   6  7
 * Schedule:                        [0, 0, i1, 0, i2, 0, i3, 0]
 */
int loop_level_into_dynamic_dimension(int level)
{
    return 1 + (level * 2 + 1);
}

/**
 * Transform the loop level into the first static schedule
 * dimension after its corresponding dynamic dimension.
 *
 * In the example below, the first static dimension that comes
 * after the corresponding dynamic dimension for
 * the loop level 0 is 3, and to 1 it is 5, ...
 *
 * Loop level               :    -1         0      1      2
 * Schedule dimension number:        0, 1   2  3   4  5   6  7
 * Schedule:                        [0, 0, i1, 0, i2, 0, i3, 0]
 */
int loop_level_into_static_dimension(int level)
{
    return loop_level_into_dynamic_dimension(level) + 1;
}

/**
 * Transform a dynamic schedule dimension into the corresponding loop level.
 *
 * In the example below, the loop level that corresponds
 * to the dynamic dimension 2 is 0, and to the dynamic dimension 4 is 1, ...
 *
 * The first dimension is the duplication dimension, the following
 * dimensions are static, dynamic, static, dynamic, ...
 *
 * Loop level               :    -1         0      1      2
 * Schedule dimension number:        0, 1   2  3   4  5   6  7
 * Schedule:                        [0, 0, i1, 0, i2, 0, i3, 0]
 */
int dynamic_dimension_into_loop_level(int dim)
{
    assert(dim % 2 == 0);
    int level = (dim - 2)/2;
    return level;
}

/**
  * Implementation internals.
  *
  * This function gets as input a loop level and translates it
  * automatically to the appropriate schedule dimension by:
  * (1) getting the dynamic schedule dimension that corresponds to
  * that loop level, then adding +1 which corresponds to the first
  * static dimension that comes after the dynamic dimension.
  *
  * Explanation of what static and dynamic dimensions are:
  * In the time-processor domain, dimensions can be either static
  * or dynamic.  Static dimensions are used to order statements
  * within a given loop level while dynamic dimensions represent
  * the actual loop levels.  For example, the computations c0 and
  * c1 in the following loop nest
  *
  * for (i=0; i<N: i++)
  *   for (j=0; j<N; j++)
  *   {
  *     c0;
  *     c1;
  *   }
  *
  * have the following representations in the iteration domain
  *
  * {c0(i,j): 0<=i<N and 0<=j<N}
  * {c1(i,j): 0<=i<N and 0<=j<N}
  *
  * and the following representation in the time-processor domain
  *
  * {c0[0,i,0,j,0]: 0<=i<N and 0<=j<N}
  * {c1[0,i,0,j,1]: 0<=i<N and 0<=j<N}
  *
  * The first dimension (dimension 0) in the time-processor
  * representation (the leftmost dimension) is a static dimension,
  * the second dimension (dimension 1) is a dynamic dimension that
  * represents the loop level i, ..., the forth dimension is a dynamic
  * dimension that represents the loop level j and the last dimension
  * (dimension 4) is a static dimension and allows the ordering of
  * c1 after c0 in the loop nest.
  *
  * \p dim has to be a static dimension, i.e. 0, 2, 4, 6, ...
  */
void computation::after_low_level(computation &comp, int level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // for loop level i return 2*i+1 which represents the
    // the static dimension just after the dynamic dimension that
    // represents the loop level i.
    int dim = loop_level_into_static_dimension(level);

    DEBUG(3, tiramisu::str_dump("Setting the schedule of ");
          tiramisu::str_dump(this->get_name());
          tiramisu::str_dump(" after ");
          tiramisu::str_dump(comp.get_name());
          tiramisu::str_dump(" at dimension ");
          tiramisu::str_dump(std::to_string(dim)));
    DEBUG(3, tiramisu::str_dump("Setting the schedule of ");
          tiramisu::str_dump(this->get_name());
          tiramisu::str_dump(" to be equal to the schedule of ");
          tiramisu::str_dump(comp.get_name());
          tiramisu::str_dump(" at all the dimensions before dimension ");
          tiramisu::str_dump(std::to_string(dim)));

    comp.get_function()->align_schedules();

    DEBUG(3, tiramisu::str_dump("Preparing to adjust the schedule of the computation ");
          tiramisu::str_dump(this->get_name()));
    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(this->get_schedule())));

    assert(this->get_schedule() != NULL);
    DEBUG(3, tiramisu::str_dump("Dimension level in which ordering dimensions will be inserted : ");
          tiramisu::str_dump(std::to_string(dim)));
    assert(dim < (signed int) isl_map_dim(this->get_schedule(), isl_dim_out));
    assert(dim >= computation::root_dimension);

    isl_map *new_sched = NULL;
    for (int i = 1; i<=dim; i=i+2)
    {
        if (i < dim)
        {
            // Get the constant in comp, add +1 to it and set it to sched1
            int order = isl_map_get_static_dim(comp.get_schedule(), i);
            new_sched = isl_map_copy(this->get_schedule());
            new_sched = add_eq_to_schedule_map(i, 0, -1, order, new_sched);
        }
        else // (i == dim)
        {
            // Get the constant in comp, add +1 to it and set it to sched1
            int order = isl_map_get_static_dim(comp.get_schedule(), i);
            new_sched = isl_map_copy(this->get_schedule());
            new_sched = add_eq_to_schedule_map(i, 0, -1, order + 10, new_sched);
        }
        this->set_schedule(new_sched);
    }

    DEBUG(3, tiramisu::str_dump("Schedule adjusted: ",
                                isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void tiramisu::buffer::set_argument_type(tiramisu::argument_t type)
{
        this->argtype = type;
}

void tiramisu::function::allocate_and_map_buffers_automatically()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(10, tiramisu::str_dump("Computing live-out computations."));
    // Compute live-in and live-out buffers
    std::vector<tiramisu::computation *> liveout = this->get_live_out_computations();
    DEBUG(10, tiramisu::str_dump("Allocating/Mapping buffers for live-out computations."));
    for (auto &comp: liveout)
        if (comp->get_automatically_allocated_buffer() == NULL)
            comp->allocate_and_map_buffer_automatically(a_output);

    DEBUG(10, tiramisu::str_dump("Computing live-in computations."));
    // Compute live-in and live-out buffers
    std::vector<tiramisu::computation *> livein =
            this->get_live_in_computations();
    DEBUG(10, tiramisu::str_dump("Allocating/Mapping buffers for live-in computations."));
    // Allocate each live-in computation that is not also live-out (we check that
    // by checking that it was not allocated yet)
    for (auto &comp: livein)
        if (comp->get_automatically_allocated_buffer() == NULL)
            comp->allocate_and_map_buffer_automatically(a_input);

    DEBUG(10, tiramisu::str_dump("Allocating/Mapping buffers for non live-in and non live-out computations."));
    // Allocate the buffers automatically for non inline computations
    // Allocate each computation that is not live-in or live-out (we check that
    // by checking that it was not allocated)
    for (int b = 0; b < this->body.size(); b++)
    {
        DEBUG(3, tiramisu::str_dump("Inline " + this->body[b]->get_name() + " " + std::to_string(this->body[b]->is_inline_computation())));
        if (this->body[b]->is_inline_computation()) {
            DEBUG(3, tiramisu::str_dump("Skipping inline computation " + this->body[b]->get_name()));
            continue;
        }
        DEBUG(10, tiramisu::str_dump("Allocating/Mapping buffers for " + this->body[b]->get_name()));
        if ((this->body[b]->get_expr().get_expr_type() == tiramisu::e_op))
        {
            if (this->body[b]->get_expr().get_op_type() != tiramisu::o_allocate)
            {
                if (this->body[b]->get_automatically_allocated_buffer() == NULL)
                    this->body[b]->allocate_and_map_buffer_automatically(a_temporary);
            }
        }
        else
        {
            if (this->body[b]->get_automatically_allocated_buffer() == NULL) {
                this->body[b]->allocate_and_map_buffer_automatically(a_temporary);
            }
        }
    }

    DEBUG_INDENT(-4);
}

tiramisu::computation *tiramisu::computation::get_first_definition()
{
	return first_definition;
}

bool tiramisu::computation::is_first_definition()
{
	return is_first;
}

bool tiramisu::computation::buffer_already_allocated()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    bool allocated = false;

    if (this->get_automatically_allocated_buffer() != NULL)
    {
	    DEBUG(3, tiramisu::str_dump("A buffer was already allocated automatically for this computation."));
	    allocated = true;;
    }
    else
    {
	    DEBUG(3, tiramisu::str_dump("No buffer was allocated automatically for this computation."));
    }

    // If this computation is not the first computation, and a buffer has
    // already been allocated for the first definition, then exit.
    if (this->has_multiple_definitions() == true)
    {
        DEBUG(3, tiramisu::str_dump("This computation has multiple definitions."));
        if (this->is_first_definition() == false)
	{
            DEBUG(3, tiramisu::str_dump("This is NOT the first definition of the computation."));
	    if (this->get_first_definition()->get_automatically_allocated_buffer() != NULL)
	    {
                DEBUG(3, tiramisu::str_dump("A buffer has already been allocated for the first computation."));
	        allocated = true;
	    }
	    else
	    {
		DEBUG(3, tiramisu::str_dump("No buffer has already been allocated for the first computation."));
		DEBUG(3, tiramisu::str_dump("Checking whether the other definitions have an automatically allocated buffer."));
	        for (auto c: this->get_first_definition()->get_updates())
		    if (c->get_automatically_allocated_buffer() != NULL)
		    {
		            DEBUG(3, tiramisu::str_dump("One of the other definitions has an automatically allocated buffer."));
			    allocated = true;
		    }
	        DEBUG(3, tiramisu::str_dump("No other definition has an automatically allocated buffer."));
	    }
	}
	else // If any of the other definitions has a buffer, exit.
	{
            DEBUG(3, tiramisu::str_dump("This is the first definition of the computation."));
            DEBUG(3, tiramisu::str_dump("Checking whether the other definitions have an automatically allocated buffer."));
	    for (auto c: this->get_updates())
		    if (c->get_automatically_allocated_buffer() != NULL)
		    {
		            DEBUG(3, tiramisu::str_dump("One of the other definitions has an automatically allocated buffer."));
			    allocated = true;
		    }
	    DEBUG(3, tiramisu::str_dump("No other definition has an automatically allocated buffer."));
	}
    }
    else
    {
        DEBUG(3, tiramisu::str_dump("This computation has only one definition."));
    }

    DEBUG_INDENT(-4);

    return allocated;
}

void tiramisu::computation::allocate_and_map_buffer_automatically(tiramisu::argument_t type)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Allocating and mapping a buffer automatically."));
    DEBUG(3, tiramisu::str_dump("Computation name: " + this->get_name()));

    // If a buffer is already allocated, exit.
    if (this->buffer_already_allocated() == true)
    {
            DEBUG(3, tiramisu::str_dump("Buffer already allocated."));
	    DEBUG_INDENT(-4);
	    return;
    }

    // If we reach this point, that means that no buffer has been allocated
    // for this computation or for the other definitions of this computation.
    std::vector<tiramisu::expr> *dim_sizes = this->compute_buffer_size();

    tiramisu::buffer *buff = NULL;

    if (this->is_first_definition() == true)
    {
        if (this->get_automatically_allocated_buffer() == NULL)
        {
	    DEBUG(3, tiramisu::str_dump("The automatically allocated buffer of this "
				        "computation is NULL."));
	    DEBUG(3, tiramisu::str_dump("Allocating an automatically allocated buffer for "
				        "this computation."));

    	    std::string buff_name;
	    buff_name = "_" + this->name + "_buffer";
	    buff = new tiramisu::buffer(buff_name,
                                (*dim_sizes),
      	                        this->get_data_type(),
                                type,
                                this->get_function());
	    this->automatically_allocated_buffer = buff;
        }
	else // automatic buffer already allocated.
		return;
    }
    else
    {
        if  (this->get_first_definition()->get_automatically_allocated_buffer() == NULL)
        {
	    DEBUG(3, tiramisu::str_dump("The automatically allocated buffer of the first "
				        "definition of this computation is NULL."));
	    DEBUG(3, tiramisu::str_dump("Allocating an automatically allocated buffer of the first "
				        "definition of this computation."));

    	    std::string buff_name;
	    buff_name = "_" + this->get_first_definition()->name + "_buffer";
	    buff = new tiramisu::buffer(buff_name,
                                (*dim_sizes),
      	                        this->get_data_type(),
                                type,
                                this->get_function());
	    this->automatically_allocated_buffer = buff;
        }
	else // first definition has an allocated array.
    	    buff = this->get_first_definition()->get_automatically_allocated_buffer();
    }

    assert(buff != NULL);

    this->automatically_allocated_buffer = buff;

    tiramisu::computation *allocation;
    if (type == tiramisu::a_temporary)
    {
        allocation = buff->allocate_at(*this, computation::root_dimension);
        allocation->set_name("_allocation_" + this->name);
        // Schedule all allocations at the beginning
        this->get_function()->automatically_allocated.push_back(allocation);
        this->get_function()->starting_computations.erase(allocation);
    }

    this->bind_to(buff);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::after(computation &comp, tiramisu::var level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(level.get_name().length() > 0);

    computation * actual_computation = this;
    while (actual_computation->is_let_stmt())
    {
        actual_computation = static_cast<constant *>(actual_computation)->get_computation_with_whom_this_is_computed();
        assert("scheduled global constant" && actual_computation != nullptr);
    }

    std::vector<int> dimensions =
	actual_computation->get_loop_level_numbers_from_dimension_names({level.get_name()});
    
    assert(dimensions.size() == 1);

    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
		level.get_name() + " is " + std::to_string(dimensions[0])));

    this->after(comp, dimensions[0]);

    DEBUG_INDENT(-4);
}
void tiramisu::computation::after(computation &comp, int level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Scheduling " + this->get_name() + " to be executed after " +
                                comp.get_name() + " at level " + std::to_string(level)));

    auto &graph = this->get_function()->sched_graph;

    auto &edges = graph[&comp];

    auto level_it = edges.find(this);

    if (level_it != edges.end())
    {
        if (level_it->second > level)
        {
            level = level_it->second;
        }
    }

    edges[this] = level;

    this->get_function()->starting_computations.erase(this);

    this->get_function()->sched_graph_reversed[this][&comp] = level;

    assert(this->get_function()->sched_graph_reversed[this].size() < 2 &&
            "Node has more than one predecessor.");

    DEBUG(10, tiramisu::str_dump("sched_graph[" + comp.get_name() + ", " +
                                 this->get_name() + "] = " + std::to_string(level)));

    DEBUG_INDENT(-4);
}

void function::dump_sched_graph_dfs(computation * comp,
                                    std::unordered_set<computation *> &visited)
{
    // Do not visit anything that was already returned
    if (visited.find(comp) != visited.end())
        return;

    visited.insert(comp);

    for (auto &edge: this->sched_graph[comp])
    {
        const std::string level = ((edge.second == computation::root_dimension) ?
                                   "root" :
                                   std::to_string(edge.second));

        DEBUG(3, tiramisu::str_dump(comp->get_unique_name() +
                                    "=[" + level + "]=>" +
                                    edge.first->get_unique_name()));

        dump_sched_graph_dfs(edge.first, visited);
    }
}

void function::dump_sched_graph()
{
    DEBUG(3, tiramisu::str_dump("Number of schedule graph roots is " +
                                std::to_string(this->starting_computations.size())));
    DEBUG(3, tiramisu::str_dump("The roots are:"));

    for (auto root: this->starting_computations)
        DEBUG(3, tiramisu::str_dump(" * " + root->get_unique_name()));

    // Contains all nodes that have been visited
    std::unordered_set<computation *> visited;

    DEBUG(3, tiramisu::str_dump("Displaying schedule graph"));

    for (auto &comp: this->starting_computations)
    {
        dump_sched_graph_dfs(comp, visited);
    }

    DEBUG(3, tiramisu::str_dump("Finished displaying schedule graph"));
}

bool function::is_sched_graph_tree_dfs(computation * comp,
                                       std::unordered_set<computation *> &visited)
{
    // Do not visit anything that was already returned
    if (visited.find(comp) != visited.end())
        return false;

    visited.insert(comp);

    for (auto &edge: this->sched_graph[comp])
    {
        if (!is_sched_graph_tree_dfs(edge.first, visited))
            return false;
    }

    return true;
}

bool function::is_sched_graph_tree()
{
    if (this->starting_computations.size() != 1)
        return false;

    // Contains all nodes that have been visited
    std::unordered_set<computation *> visited;

    for (auto &comp: this->starting_computations)
    {
        if (!is_sched_graph_tree_dfs(comp, visited))
            return false;
    }

    return true;
}


void function::gen_ordering_schedules()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if (this->use_low_level_scheduling_commands)
    {
        DEBUG(3, tiramisu::str_dump("Low level scheduling commands were used."));
        DEBUG(3, tiramisu::str_dump("Discarding high level scheduling commands."));
        return;
    }

    this->dump_sched_graph();

    if(this->is_sched_graph_tree())
    {
	std::priority_queue<int> level_to_check;
	std::unordered_map<int, std::deque<computation *>> level_queue;

	auto current_comp = *(this->starting_computations.begin());

	auto init_sched = automatically_allocated;
	init_sched.push_back(current_comp);

	for (auto it = init_sched.begin(); it != init_sched.end() && it + 1 != init_sched.end(); it++)
	    (*(it+1))->after_low_level(**it, computation::root_dimension);

	bool comps_remain = true;
	while(comps_remain)
	{
	    for (auto &edge: this->sched_graph[current_comp])
	    {
		if (level_queue[edge.second].size() == 0)
		    level_to_check.push(edge.second);

		level_queue[edge.second].push_back(edge.first);
	    }

	    comps_remain = level_to_check.size() > 0;
	    // If we haven't exhausted all computations
	    if (comps_remain)
	    {
		int fuse_level = level_to_check.top();
		auto next_comp = level_queue[fuse_level].front();
		level_queue[fuse_level].pop_front();

		// assert(this->get_max_iteration_domains_dim() > fuse_level);

		next_comp->after_low_level((*current_comp), fuse_level);

		current_comp = next_comp;
		if (level_queue[fuse_level].size() == 0)
		    level_to_check.pop();
	    }
	}
    }
}

void computation::before(computation &comp, tiramisu::var dim)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    comp.after(*this, dim);

    DEBUG_INDENT(-4);
}

void computation::before(computation &comp, int dim)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    comp.after(*this, dim);

    DEBUG_INDENT(-4);
}

void computation::between(computation &before_c, tiramisu::var before_dim_var, computation &after_c, tiramisu::var after_dim_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(before_dim_var.get_name().length() > 0);
    assert(after_dim_var.get_name().length() > 0);

    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({before_dim_var.get_name(), after_dim_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int before_dim = dimensions[0];
    int after_dim = dimensions[1];

    DEBUG(3, tiramisu::str_dump("Scheduling " + this->get_name() + " between " +
                                before_c.get_name() + " and " + after_c.get_name()));

    auto f = this->get_function();

    if (f->sched_graph[&before_c].find(&after_c) != f->sched_graph[&before_c].end()) {
        DEBUG(3, tiramisu::str_dump("Removing pre-existing edge"));
        f->sched_graph[&before_c].erase(&after_c);
        f->sched_graph_reversed[&after_c].erase(&before_c);
    }

    this->after(before_c, before_dim);
    after_c.after(*this, after_dim);

    DEBUG_INDENT(-4);
}

void computation::gpu_tile(tiramisu::var L0_var, tiramisu::var L1_var, int sizeX, int sizeY)
{
    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);

    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(),
							   L1_var.get_name()});

    assert(dimensions.size() == 2);

    int L0 = dimensions[0];
    int L1 = dimensions[1];

    this->check_dimensions_validity({L0, L1});

    assert(L0 >= 0);
    assert(L1 >= 0);
    assert((L1 == L0 + 1));
    assert(sizeX > 0);
    assert(sizeY > 0);

    this->tile(L0, L1, sizeX, sizeY);
    this->tag_gpu_block_level(L0, L1);
    this->tag_gpu_thread_level(L0 + 2, L1 + 2);
}

void computation::gpu_tile(tiramisu::var L0_var, tiramisu::var L1_var, tiramisu::var L2_var, int sizeX, int sizeY, int sizeZ)
{
    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(L2_var.get_name().length() > 0);

    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(),
							   L1_var.get_name(),
							   L2_var.get_name()});

    assert(dimensions.size() == 3);

    int L0 = dimensions[0];
    int L1 = dimensions[1];
    int L2 = dimensions[2];

    this->check_dimensions_validity({L0, L1, L2});

    assert((L1 == L0 + 1));
    assert((L2 == L1 + 1));
    assert(sizeX > 0);
    assert(sizeY > 0);
    assert(sizeZ > 0);

    this->tile(L0, L1, L2, sizeX, sizeY, sizeZ);
    this->tag_gpu_block_level(L0, L1, L2);
    this->tag_gpu_thread_level(L0 + 3, L1 + 3, L2 + 3);
}

void computation::assert_names_not_assigned(
	std::vector<std::string> dimensions)
{
    for (auto const dim: dimensions)
    {
	int d = isl_map_find_dim_by_name(this->get_schedule(), isl_dim_out,
			dim.c_str());
	if (d >= 0)
	    tiramisu::error("Dimension " + dim + " is already in use.", true);

	d = isl_map_find_dim_by_name(this->get_schedule(), isl_dim_in,
			dim.c_str());
	if (d >= 0)
	    tiramisu::error("Dimension " + dim + " is already in use.", true);
    }
}

void computation::check_dimensions_validity(std::vector<int> dimensions)
{
    assert(dimensions.size() > 0);

    for (auto const dim: dimensions)
    {
	DEBUG(10, tiramisu::str_dump("Checking the validity of loop level " +
				     std::to_string(dim)));

	assert(dim >= computation::root_dimension);

	if (loop_level_into_dynamic_dimension(dim) >=
		isl_space_dim(isl_map_get_space(this->get_schedule()),
			      isl_dim_out))
	{
	    tiramisu::error("The dynamic dimension " +
		std::to_string(loop_level_into_dynamic_dimension(dim)) +
		" is not less than the number of dimensions of the "
		"time-space domain " +
		std::to_string(isl_space_dim(isl_map_get_space(
				this->get_schedule()), isl_dim_out)) , true);
	}
    }
}

void computation::set_loop_level_names(std::vector<std::string> names)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(names.size() > 0);

    for (int i = 0; i < this->get_loop_levels_number(); i++)
    {
	if (isl_map_has_dim_name(this->get_schedule(), isl_dim_out, loop_level_into_dynamic_dimension(i)) == isl_bool_true)
	{
	    this->schedule = isl_map_set_dim_name(this->get_schedule(),
	        isl_dim_out,
		loop_level_into_dynamic_dimension(i),
                names[i].c_str());
  	    DEBUG(3, tiramisu::str_dump("Setting the name of loop level " + std::to_string(i) + " into " + names[i].c_str()));
	}
    }

    DEBUG(3, tiramisu::str_dump("The schedule after renaming: ", isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::set_schedule_domain_dim_names(std::vector<int> loop_levels,
	std::vector<std::string> names)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    this->check_dimensions_validity(loop_levels);
    assert(names.size() > 0);
    assert(names.size() == loop_levels.size());

    for (int i = 0; i < loop_levels.size(); i++)
    {
	assert(loop_levels[i] <= isl_map_dim(this->get_schedule(), isl_dim_in));
	this->schedule = isl_map_set_dim_name(this->get_schedule(),
			    isl_dim_in, loop_levels[i], names[i].c_str());
  	DEBUG(3, tiramisu::str_dump("Setting the name of the domain of the schedule dimension " + std::to_string(loop_levels[i]) + " into " + names[i].c_str()));
    }

    DEBUG(3, tiramisu::str_dump("The schedule after renaming: ", isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::set_loop_level_names(std::vector<int> loop_levels,
	std::vector<std::string> names)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    this->check_dimensions_validity(loop_levels);
    assert(names.size() > 0);
    assert(names.size() == loop_levels.size());

    for (int i = 0; i < loop_levels.size(); i++)
    {
	if (loop_level_into_static_dimension(loop_levels[i]) <= isl_map_dim(this->get_schedule(), isl_dim_out))
	{
	    this->schedule = isl_map_set_dim_name(this->get_schedule(),
	        isl_dim_out,
		loop_level_into_dynamic_dimension(loop_levels[i]),
                names[i].c_str());
  	    DEBUG(3, tiramisu::str_dump("Setting the name of loop level " + std::to_string(loop_levels[i]) + " into " + names[i].c_str()));
	}
    }

    DEBUG(3, tiramisu::str_dump("The schedule after renaming: ", isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::tile(int L0, int L1, int sizeX, int sizeY)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Check that the two dimensions are consecutive.
    // Tiling only applies on a consecutive band of loop dimensions.
    assert(L1 == L0 + 1);
    assert((sizeX > 0) && (sizeY > 0));
    assert(this->get_iteration_domain() != NULL);
    this->check_dimensions_validity({L0, L1});

//    this->separateAndSplit(L0, sizeX);
//    this->separateAndSplit(L1 + 1, sizeY);
    this->split(L0, sizeX);
    this->split(L1 + 1, sizeY);

    this->interchange(L0 + 1, L1 + 1);

    DEBUG_INDENT(-4);
}

std::vector<int> computation::get_loop_level_numbers_from_dimension_names(
	std::vector<std::string> dim_names)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(dim_names.size() > 0);

    std::vector<int> dim_numbers;

    for (auto const dim: dim_names)
    {
	assert(dim.size()>0);

	DEBUG(10, tiramisu::str_dump("Searching for the dimension " + dim));

	if (dim == "root")
	{
	    int d = computation::root_dimension;
	    dim_numbers.push_back(d);
	}
	else
	{
	    int d = isl_map_find_dim_by_name(this->get_schedule(), isl_dim_out,
			dim.c_str());
	    DEBUG(10, tiramisu::str_dump("Searching in the range of ",
					isl_map_to_str(this->get_schedule())));

	    if (d < 0)
		tiramisu::error("Dimension " + dim + " not found.", true);

	    DEBUG(10, tiramisu::str_dump("Corresponding loop level is " +
			std::to_string(dynamic_dimension_into_loop_level(d))));

	    dim_numbers.push_back(dynamic_dimension_into_loop_level(d));
	}
    }

    this->check_dimensions_validity(dim_numbers);

    DEBUG_INDENT(-4);

    return dim_numbers;
}

void computation::name_unnamed_time_space_dimensions()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_map *sched = this->get_schedule();

    assert(sched != NULL);

    for (int i = 0; i < this->get_loop_levels_number(); i++)
    {
	if (isl_map_has_dim_name(sched, isl_dim_out, loop_level_into_dynamic_dimension(i)) == isl_bool_false)
	    sched = isl_map_set_dim_name(sched, isl_dim_out, loop_level_into_dynamic_dimension(i), generate_new_variable_name().c_str());
    }

    this->set_schedule(sched);

    DEBUG_INDENT(-4);
}

void computation::name_unnamed_iteration_domain_dimensions()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_set *iter = this->get_iteration_domain();

    assert(iter != NULL);

    for (int i = 0; i < this->get_iteration_domain_dimensions_number(); i++)
    {
	if (isl_set_has_dim_name(iter, isl_dim_set, i) == isl_bool_false)
	    iter = isl_set_set_dim_name(iter, isl_dim_set, i,
			generate_new_variable_name().c_str());
    }

    this->set_iteration_domain(iter);

    DEBUG_INDENT(-4);
}

std::vector<std::string> computation::get_iteration_domain_dimension_names()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_set *iter = this->get_iteration_domain();

    assert(iter != NULL);

    std::vector<std::string> result;

    for (int i = 0; i < this->get_iteration_domain_dimensions_number(); i++)
    {
	if (isl_set_has_dim_name(iter, isl_dim_set, i))
	    result.push_back(std::string(isl_set_get_dim_name(iter,
					    isl_dim_set, i)));
	else
	    tiramisu::error("All iteration domain dimensions must have "
		"a name.", true);
    }

    assert(result.size() == this->get_iteration_domain_dimensions_number());

    DEBUG_INDENT(-4);

    return result;
}

void computation::tile(tiramisu::var L0, tiramisu::var L1,
	tiramisu::var L2, int sizeX, int sizeY, int sizeZ)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0.get_name().length() > 0);
    assert(L1.get_name().length() > 0);
    assert(L2.get_name().length() > 0);

    tiramisu::var L0_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L1_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L2_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L0_inner = tiramisu::var(generate_new_variable_name());
    tiramisu::var L1_inner = tiramisu::var(generate_new_variable_name());
    tiramisu::var L2_inner = tiramisu::var(generate_new_variable_name());

    this->tile(L0, L1, L2, sizeX, sizeY, sizeZ,
		L0_outer, L1_outer, L0_outer, L0_inner, L1_inner, L2_inner);

    DEBUG_INDENT(-4);
}

void computation::tile(tiramisu::var L0, tiramisu::var L1,
	int sizeX, int sizeY)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0.get_name().length() > 0);
    assert(L1.get_name().length() > 0);

    tiramisu::var L0_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L1_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L0_inner = tiramisu::var(generate_new_variable_name());
    tiramisu::var L1_inner = tiramisu::var(generate_new_variable_name());

    this->tile(L0, L1, sizeX, sizeY,
		L0_outer, L1_outer, L0_inner, L1_inner);

    DEBUG_INDENT(-4);
}

void computation::tile(tiramisu::var L0, tiramisu::var L1, tiramisu::var L2,
	int sizeX, int sizeY, int sizeZ,
	tiramisu::var L0_outer, tiramisu::var L1_outer,
	tiramisu::var L2_outer, tiramisu::var L0_inner,
	tiramisu::var L1_inner, tiramisu::var L2_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0.get_name().length() > 0);
    assert(L1.get_name().length() > 0);
    assert(L2.get_name().length() > 0);
    assert(L0_outer.get_name().length() > 0);
    assert(L1_outer.get_name().length() > 0);
    assert(L2_outer.get_name().length() > 0);
    assert(L0_inner.get_name().length() > 0);
    assert(L1_inner.get_name().length() > 0);
    assert(L2_inner.get_name().length() > 0);

    this->assert_names_not_assigned({L0_outer.get_name(), L1_outer.get_name(),
				    L2_outer.get_name(), L0_inner.get_name(),
				    L1_inner.get_name(), L2_inner.get_name()});

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0.get_name(),
							   L1.get_name(),
							   L2.get_name()});
    assert(dimensions.size() == 3);

    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
		L0.get_name() + " is " + std::to_string(dimensions[0])));
    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
		L1.get_name() + " is " + std::to_string(dimensions[1])));
    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
		L2.get_name() + " is " + std::to_string(dimensions[2])));

    this->tile(dimensions[0], dimensions[1], dimensions[2],
		sizeX, sizeY, sizeZ);

    this->update_names(original_loop_level_names, {L0_outer.get_name(), L1_outer.get_name(), L2_outer.get_name(),
						   L0_inner.get_name(), L1_inner.get_name(), L2_inner.get_name()}, dimensions[0], 3);

    DEBUG_INDENT(-4);
}

void computation::tile(tiramisu::var L0, tiramisu::var L1,
      int sizeX, int sizeY,
      tiramisu::var L0_outer, tiramisu::var L1_outer,
      tiramisu::var L0_inner, tiramisu::var L1_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0.get_name().length() > 0);
    assert(L1.get_name().length() > 0);
    assert(L0_outer.get_name().length() > 0);
    assert(L1_outer.get_name().length() > 0);
    assert(L0_inner.get_name().length() > 0);
    assert(L1_inner.get_name().length() > 0);

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    this->assert_names_not_assigned({L0_outer.get_name(), L1_outer.get_name(),
				    L0_inner.get_name(), L1_inner.get_name()});

    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0.get_name(),
							   L1.get_name()});
    assert(dimensions.size() == 2);

    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
		L0.get_name() + " is " + std::to_string(dimensions[0])));
    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
		L1.get_name() + " is " + std::to_string(dimensions[1])));

    this->tile(dimensions[0], dimensions[1], sizeX, sizeY);

    // Replace the original dimension name with two new dimension names
    this->update_names(original_loop_level_names, {L0_outer.get_name(), L1_outer.get_name(), L0_inner.get_name(), L1_inner.get_name()}, dimensions[0], 2);

    DEBUG_INDENT(-4);
}

void computation::tile(int L0, int L1, int L2, int sizeX, int sizeY, int sizeZ)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Check that the two dimensions are consecutive.
    // Tiling only applies on a consecutive band of loop dimensions.
    assert(L1 == L0 + 1);
    assert(L2 == L1 + 1);
    assert((sizeX > 0) && (sizeY > 0) && (sizeZ > 0));
    assert(this->get_iteration_domain() != NULL);

    this->check_dimensions_validity({L0, L1, L2});

    //  Original loops
    //  L0
    //    L1
    //      L2

    this->split(L0, sizeX); // Split L0 into L0 and L0_prime
    // Compute the new L1 and the new L2 and the newly created L0 (called L0 prime)
    int L0_prime = L0 + 1;
    L1 = L1 + 1;
    L2 = L2 + 1;

    //  Loop after transformation
    //  L0
    //    L0_prime
    //      L1
    //        L2

    this->split(L1, sizeY);
    int L1_prime = L1 + 1;
    L2 = L2 + 1;

    //  Loop after transformation
    //  L0
    //    L0_prime
    //      L1
    //        L1_prime
    //          L2

    this->split(L2, sizeZ);

    //  Loop after transformation
    //  L0
    //    L0_prime
    //      L1
    //        L1_prime
    //          L2
    //            L2_prime

    this->interchange(L0_prime, L1);
    // Change the position of L0_prime to the new position
    int temp = L0_prime;
    L0_prime = L1;
    L1 = temp;

    //  Loop after transformation
    //  L0
    //    L1
    //      L0_prime
    //        L1_prime
    //          L2
    //            L2_prime

    this->interchange(L0_prime, L2);
    // Change the position of L0_prime to the new position
    temp = L0_prime;
    L0_prime = L2;
    L2 = temp;

    //  Loop after transformation
    //  L0
    //    L1
    //      L2
    //        L1_prime
    //          L0_prime
    //            L2_prime

    this->interchange(L1_prime, L0_prime);

    //  Loop after transformation
    //  L0
    //    L1
    //      L2
    //        L0_prime
    //          L1_prime
    //            L2_prime

    DEBUG_INDENT(-4);
}


void computation::interchange(tiramisu::var L0_var, tiramisu::var L1_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];

    this->interchange(L0, L1);

    DEBUG_INDENT(-4);
}

/**
 * This function modifies the schedule of the computation so that the two loop
 * levels L0 and L1 are interchanged (swapped).
 */
void computation::interchange(int L0, int L1)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int inDim0 = loop_level_into_dynamic_dimension(L0);
    int inDim1 = loop_level_into_dynamic_dimension(L1);

    assert(inDim0 >= 0);
    assert(inDim0 < isl_space_dim(isl_map_get_space(this->get_schedule()),
                                  isl_dim_out));
    assert(inDim1 >= 0);
    assert(inDim1 < isl_space_dim(isl_map_get_space(this->get_schedule()),
                                  isl_dim_out));

    isl_map *schedule = this->get_schedule();

    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(schedule)));
    DEBUG(3, tiramisu::str_dump("Interchanging the dimensions " + std::to_string(
                                    L0) + " and " + std::to_string(L1)));

    int n_dims = isl_map_dim(schedule, isl_dim_out);

    std::string inDim0_str = isl_map_get_dim_name(schedule, isl_dim_out, inDim0);
    std::string inDim1_str = isl_map_get_dim_name(schedule, isl_dim_out, inDim1);

    std::vector<isl_id *> dimensions;

    // ------------------------------------------------------------
    // Create a map for the duplicate schedule.
    // ------------------------------------------------------------

    std::string map = "{ " + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            int duplicate_ID = isl_map_get_static_dim(schedule, 0);
            map = map + std::to_string(duplicate_ID);
        }
        else
        {
            if (isl_map_get_dim_name(schedule, isl_dim_out, i) == NULL)
            {
                isl_id *new_id = isl_id_alloc(this->get_ctx(), generate_new_variable_name().c_str(), NULL);
                schedule = isl_map_set_dim_id(schedule, isl_dim_out, i, new_id);
            }

            map = map + isl_map_get_dim_name(schedule, isl_dim_out, i);
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] ->" + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            int duplicate_ID = isl_map_get_static_dim(schedule, 0);
            map = map + std::to_string(duplicate_ID);
        }
        else
        {
            if ((i != inDim0) && (i != inDim1))
            {
                map = map + isl_map_get_dim_name(schedule, isl_dim_out, i);
                dimensions.push_back(isl_map_get_dim_id(schedule, isl_dim_out, i));
            }
            else if (i == inDim0)
            {
                map = map + inDim1_str;
                isl_id *id1 = isl_id_alloc(this->get_ctx(), inDim1_str.c_str(), NULL);
                dimensions.push_back(id1);
            }
            else if (i == inDim1)
            {
                map = map + inDim0_str;
                isl_id *id1 = isl_id_alloc(this->get_ctx(), inDim0_str.c_str(), NULL);
                dimensions.push_back(id1);
            }
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "]}";

    DEBUG(3, tiramisu::str_dump("A map that transforms the duplicate"));
    DEBUG(3, tiramisu::str_dump(map.c_str()));

    isl_map *transformation_map = isl_map_read_from_str(this->get_ctx(), map.c_str());


    transformation_map = isl_map_set_tuple_id(
                             transformation_map, isl_dim_in, isl_map_get_tuple_id(isl_map_copy(schedule), isl_dim_out));
    isl_id *id_range = isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL);
    transformation_map = isl_map_set_tuple_id(
                             transformation_map, isl_dim_out, id_range);


    // Check that the names of each dimension is well set
    for (int i = 1; i < isl_map_dim(transformation_map, isl_dim_in); i++)
    {
        isl_id *dim_id = isl_id_copy(dimensions[i - 1]);
        transformation_map = isl_map_set_dim_id(transformation_map, isl_dim_out, i, dim_id);
        assert(isl_map_has_dim_name(transformation_map, isl_dim_in, i));
        assert(isl_map_has_dim_name(transformation_map, isl_dim_out, i));
    }

    DEBUG(3, tiramisu::str_dump("Final transformation map : ", isl_map_to_str(transformation_map)));

    schedule = isl_map_apply_range(isl_map_copy(schedule), isl_map_copy(transformation_map));

    DEBUG(3, tiramisu::str_dump("Schedule after interchange: ", isl_map_to_str(schedule)));

    this->set_schedule(schedule);

    DEBUG_INDENT(-4);
}

/**
 * Get a map as input.  Go through all the basic maps, keep only
 * the basic map of the duplicate ID.
 */
isl_map *isl_map_filter_bmap_by_dupliate_ID(int ID, isl_map *map)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_map *identity = isl_map_universe(isl_map_get_space(map));
    identity = isl_set_identity(isl_map_range(isl_map_copy(map)));
    DEBUG(3, tiramisu::str_dump("Identity created from the range of the map: ",
                                isl_map_to_str(identity)));

    identity = isl_map_set_const_dim(identity, 0, ID);

    return isl_map_apply_range(isl_map_copy(map), identity);

    DEBUG_INDENT(-4);
}

/**
 * domain_constraints_set: a set defined on the space of the domain of the
 * schedule.
 *
 * range_constraints_set: a set defined on the space of the range of the
 * schedule.
 */
tiramisu::computation *computation::duplicate(std::string domain_constraints_set,
        std::string range_constraints_set)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);


    DEBUG(3, tiramisu::str_dump("Creating a schedule that duplicates ");
          tiramisu::str_dump(this->get_name()););
    DEBUG(3, tiramisu::str_dump("The duplicate is defined with the following constraints on the domain of the schedule: ");
          tiramisu::str_dump(domain_constraints_set));
    DEBUG(3, tiramisu::str_dump("and the following constraints on the range of the schedule: ");
          tiramisu::str_dump(range_constraints_set));

    this->get_function()->align_schedules();


    DEBUG(3, tiramisu::str_dump("Preparing to adjust the schedule of the computation ");
          tiramisu::str_dump(this->get_name()));
    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(this->get_schedule())));

    assert(this->get_schedule() != NULL);

    DEBUG(3, tiramisu::str_dump("The ID of the last duplicate of this computation (i.e., number of duplicates) is : "
                                + std::to_string(this->get_duplicates_number())));

    DEBUG(3, tiramisu::str_dump("Now creating a map for the new duplicate."));
    int new_ID = this->get_duplicates_number() + 1;
    this->duplicate_number++; // Increment the duplicate number.
    isl_map *new_sched = isl_map_copy(this->get_schedule());
    DEBUG(3, tiramisu::str_dump("The map of the original: ", isl_map_to_str(new_sched)));

    // Intersecting the range of the schedule with the domain and range provided by the user.
    isl_set *domain_set = NULL;
    if (domain_constraints_set.length() > 0)
    {
        domain_set = isl_set_read_from_str(this->get_ctx(), domain_constraints_set.c_str());
    }
    isl_set *range_set = NULL;
    if (range_constraints_set.length() > 0)
    {
        range_set = isl_set_read_from_str(this->get_ctx(), range_constraints_set.c_str());
    }

    if (domain_set != NULL)
    {
        DEBUG(3, tiramisu::str_dump("Intersecting the following schedule and set on the domain."));
        DEBUG(3, tiramisu::str_dump("Schedule: ", isl_map_to_str(new_sched)));
        DEBUG(3, tiramisu::str_dump("Set: ", isl_set_to_str(domain_set)));

        new_sched = isl_map_intersect_domain(new_sched, domain_set);
    }

    if (range_set != NULL)
    {
        DEBUG(3, tiramisu::str_dump("Intersecting the following schedule and set on the range."));
        DEBUG(3, tiramisu::str_dump("Schedule: ", isl_map_to_str(new_sched)));
        DEBUG(3, tiramisu::str_dump("Set: ", isl_set_to_str(range_set)));

        new_sched = isl_map_intersect_range(new_sched, range_set);
    }

    new_sched = this->simplify(new_sched);
    DEBUG(3, tiramisu::str_dump("Resulting schedule: ", isl_map_to_str(new_sched)));


    // Setting the duplicate dimension
    new_sched = isl_map_set_const_dim(new_sched, 0, new_ID);
    DEBUG(3, tiramisu::str_dump("After setting the dimension 0 to the new_ID: ",
                                isl_map_to_str(new_sched)));
    DEBUG(3, tiramisu::str_dump("The map of the new duplicate is now: ", isl_map_to_str(new_sched)));

    // Create the duplicate computation.
    tiramisu::computation *new_c = this->copy();
    new_c->set_schedule(isl_map_copy(new_sched));

    DEBUG(3, tiramisu::str_dump("The schedule of the original computation: "));
    isl_map_dump(this->get_schedule());
    DEBUG(3, tiramisu::str_dump("The schedule of the duplicate: "));
    isl_map_dump(new_c->get_schedule());

    DEBUG_INDENT(-4);

    return new_c;
}

// TODO: fix this function
isl_map *add_eq_to_schedule_map(int dim0, int in_dim_coefficient, int out_dim_coefficient,
                                int const_conefficient, isl_map *sched)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("The schedule :", isl_map_to_str(sched)));
    DEBUG(3, tiramisu::str_dump("Editing the dimension " + std::to_string(dim0)));
    DEBUG(3, tiramisu::str_dump("Coefficient of the input dimension " + std::to_string(
                                    in_dim_coefficient)));
    DEBUG(3, tiramisu::str_dump("Coefficient of the output dimension " + std::to_string(
                                    out_dim_coefficient)));
    DEBUG(3, tiramisu::str_dump("Coefficient of the constant " + std::to_string(const_conefficient)));

    isl_map *identity = isl_set_identity(isl_map_range(isl_map_copy(sched)));
    identity = isl_map_universe(isl_map_get_space(identity));
    isl_space *sp = isl_map_get_space(identity);
    isl_local_space *lsp = isl_local_space_from_space(isl_space_copy(sp));

    // Create a transformation map that transforms the schedule.
    for (int i = 0; i < isl_map_dim (identity, isl_dim_out); i++)
        if (i == dim0)
        {
            isl_constraint *cst = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, dim0, in_dim_coefficient);
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, dim0, -out_dim_coefficient);
            // TODO: this should be inverted into const_conefficient.
            cst = isl_constraint_set_constant_si(cst, -const_conefficient);
            identity = isl_map_add_constraint(identity, cst);
            DEBUG(3, tiramisu::str_dump("Setting the constraint for dimension " + std::to_string(dim0)));
            DEBUG(3, tiramisu::str_dump("The identity schedule is now: ", isl_map_to_str(identity)));
        }
        else
        {
            // Set equality constraints for dimensions
            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
            identity = isl_map_add_constraint(identity, cst2);
        }

    isl_map *final_identity = identity;
    DEBUG(3, tiramisu::str_dump("The transformation map is: ", isl_map_to_str(final_identity)));
    sched = isl_map_apply_range (sched, final_identity);
    DEBUG(3, tiramisu::str_dump("The schedule after being transformed: ", isl_map_to_str(sched)));

    DEBUG_INDENT(-4);

    return sched;
}

isl_map *add_ineq_to_schedule_map(int duplicate_ID, int dim0, int in_dim_coefficient,
                                  int out_dim_coefficient, int const_conefficient, isl_map *sched)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Editing the duplicate " + std::to_string(
                                    duplicate_ID) + " of the schedule :", isl_map_to_str(sched)));
    DEBUG(3, tiramisu::str_dump("Editing the dimension " + std::to_string(dim0)));
    DEBUG(3, tiramisu::str_dump("Coefficient of the input dimension " + std::to_string(
                                    in_dim_coefficient)));
    DEBUG(3, tiramisu::str_dump("Coefficient of the output dimension " + std::to_string(
                                    out_dim_coefficient)));
    DEBUG(3, tiramisu::str_dump("Coefficient of the constant " + std::to_string(const_conefficient)));

    isl_map *identity = isl_set_identity(isl_map_range(isl_map_copy(sched)));
    identity = isl_map_universe(isl_map_get_space(identity));
    isl_space *sp = isl_map_get_space(identity);
    isl_local_space *lsp = isl_local_space_from_space(isl_space_copy(sp));

    // Create a transformation map that applies only on the map that have
    // duplicate_ID as an ID.
    for (int i = 0; i < isl_map_dim (identity, isl_dim_out); i++)
        if (i == 0)
        {
            isl_constraint *cst = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, 0, 1);
            cst = isl_constraint_set_constant_si(cst, -duplicate_ID);
            identity = isl_map_add_constraint(identity, cst);

            // Set equality constraints for the first dimension (to keep the value of the duplicate ID)
            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
            identity = isl_map_add_constraint(identity, cst2);

            DEBUG(3, tiramisu::str_dump("Setting the constant " + std::to_string(
                                            duplicate_ID) + " for dimension 0."));
        }
        else if (i == dim0)
        {
            isl_constraint *cst = isl_constraint_alloc_inequality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, dim0, in_dim_coefficient);
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, dim0, -out_dim_coefficient);
            cst = isl_constraint_set_constant_si(cst, -const_conefficient);
            identity = isl_map_add_constraint(identity, cst);
            DEBUG(3, tiramisu::str_dump("Setting the constraint for dimension " + std::to_string(dim0)));
            DEBUG(3, tiramisu::str_dump("The identity schedule is now: ", isl_map_to_str(identity)));
        }
        else
        {
            // Set equality constraints for dimensions
            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
            identity = isl_map_add_constraint(identity, cst2);
        }

    isl_map *final_identity = identity;
    DEBUG(3, tiramisu::str_dump("The identity schedule is now: ", isl_map_to_str(final_identity)));

    isl_map *identity2;

    // Now set map that keep schedules of the other duplicates without any modification.
    DEBUG(3, tiramisu::str_dump("Setting a map to keep the schedules of the other duplicates that have an ID > this duplicate"));
    identity2 = isl_set_identity(isl_map_range(isl_map_copy(sched)));
    identity2 = isl_map_universe(isl_map_get_space(identity2));
    sp = isl_map_get_space(identity2);
    lsp = isl_local_space_from_space(isl_space_copy(sp));
    for (int i = 0; i < isl_map_dim (identity2, isl_dim_out); i++)
    {
        if (i == 0)
        {
            isl_constraint *cst = isl_constraint_alloc_inequality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, 0, 1);
            cst = isl_constraint_set_constant_si(cst, -duplicate_ID - 1);
            identity2 = isl_map_add_constraint(identity2, cst);
        }
        isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
        cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
        cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
        identity2 = isl_map_add_constraint(identity2, cst2);
    }

    DEBUG(3, tiramisu::str_dump("The identity schedule is now: ", isl_map_to_str(identity2)));
    final_identity = isl_map_union (final_identity, identity2);

    if (duplicate_ID > 0)
    {
        DEBUG(3, tiramisu::str_dump("Setting a map to keep the schedules of the other duplicates that have an ID < this duplicate"));
        identity2 = isl_set_identity(isl_map_range(isl_map_copy(sched)));
        identity2 = isl_map_universe(isl_map_get_space(identity2));
        sp = isl_map_get_space(identity2);
        lsp = isl_local_space_from_space(isl_space_copy(sp));
        for (int i = 0; i < isl_map_dim (identity2, isl_dim_out); i++)
        {
            if (i == 0)
            {
                isl_constraint *cst = isl_constraint_alloc_inequality(isl_local_space_copy(lsp));
                cst = isl_constraint_alloc_inequality(isl_local_space_copy(lsp));
                cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, 0, -1);
                cst = isl_constraint_set_constant_si(cst, duplicate_ID - 1);
                identity2 = isl_map_add_constraint(identity2, cst);
            }
            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
            identity2 = isl_map_add_constraint(identity2, cst2);
        }
        DEBUG(3, tiramisu::str_dump("The identity schedule is now: ", isl_map_to_str(identity2)));
        final_identity = isl_map_union (final_identity, identity2);
    }

    DEBUG(3, tiramisu::str_dump("The final transformation map is: ", isl_map_to_str(final_identity)));
    sched = isl_map_apply_range (sched, final_identity);
    DEBUG(3, tiramisu::str_dump("The schedule after being modified: ", isl_map_to_str(sched)));

    DEBUG_INDENT(-4);

    return sched;
}

void computation::shift(tiramisu::var L0_var, int n)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    this->shift(L0, n);

    DEBUG_INDENT(-4);
}

void computation::shift(int L0, int n)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int dim0 = loop_level_into_dynamic_dimension(L0);

    assert(this->get_schedule() != NULL);
    assert(dim0 >= 0);
    assert(dim0 < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));


    DEBUG(3, tiramisu::str_dump("Creating a schedule that shifts the loop level ");
          tiramisu::str_dump(std::to_string(L0));
          tiramisu::str_dump(" of the computation ");
          tiramisu::str_dump(this->get_name());
          tiramisu::str_dump(" by ");
          tiramisu::str_dump(std::to_string(n)));

    this->get_function()->align_schedules();
    assert(this->get_schedule() != NULL);

    DEBUG(3, tiramisu::str_dump("Original schedule: ",
                                isl_map_to_str(this->get_schedule())));

    isl_map *new_sched = isl_map_copy(this->get_schedule());
    new_sched = add_eq_to_schedule_map(dim0, -1, -1, n, new_sched);
    this->set_schedule(new_sched);
    DEBUG(3, tiramisu::str_dump("Schedule after shifting: ",
                                isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

isl_set *computation::simplify(isl_set *set)
{
    set = this->intersect_set_with_context(set);
    set = isl_set_coalesce(set);
    set = isl_set_remove_redundancies(set);

    return set;
}

isl_map *computation::simplify(isl_map *map)
{
    map = this->intersect_map_domain_with_context(map);
    map = isl_map_coalesce(map);

    return map;
}

isl_set *computation::intersect_set_with_context(isl_set *set)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Unify the space of the context and the "missing" set so that we can intersect them.
    isl_set *context = isl_set_copy(this->get_function()->get_program_context());
    if (context != NULL)
    {
        isl_space *model = isl_set_get_space(isl_set_copy(context));
        set = isl_set_align_params(set, isl_space_copy(model));
        DEBUG(10, tiramisu::str_dump("Context: ", isl_set_to_str(context)));
        DEBUG(10, tiramisu::str_dump("Set after aligning its parameters with the context parameters: ",
                                     isl_set_to_str (set)));

        isl_id *missing_id1 = NULL;
        if (isl_set_has_tuple_id(set) == isl_bool_true)
        {
            missing_id1 = isl_set_get_tuple_id(set);
        }
        else
        {
            std::string name = isl_set_get_tuple_name(set);
            assert(name.size() > 0);
            missing_id1 = isl_id_alloc(this->get_ctx(), name.c_str(), NULL);
        }

        int nb_dims = isl_set_dim(set, isl_dim_set);
        context = isl_set_add_dims(context, isl_dim_set, nb_dims);
        DEBUG(10, tiramisu::str_dump("Context after adding dimensions to make it have the same number of dimensions as missing: ",
                                     isl_set_to_str (context)));
        context = isl_set_set_tuple_id(context, isl_id_copy(missing_id1));
        DEBUG(10, tiramisu::str_dump("Context after setting its tuple ID to be equal to the tuple ID of missing: ",
                                     isl_set_to_str (context)));
        set = isl_set_intersect(set, isl_set_copy(context));
        DEBUG(10, tiramisu::str_dump("Set after intersecting with the program context: ",
                                     isl_set_to_str (set)));
    }

    DEBUG_INDENT(-4);

    return set;
}

isl_map *computation::intersect_map_domain_with_context(isl_map *map)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Unify the space of the context and the "missing" set so that we can intersect them.
    isl_set *context = isl_set_copy(this->get_function()->get_program_context());
    if (context != NULL)
    {
        isl_space *model = isl_set_get_space(isl_set_copy(context));
        map = isl_map_align_params(map, isl_space_copy(model));
        DEBUG(10, tiramisu::str_dump("Context: ", isl_set_to_str(context)));
        DEBUG(10, tiramisu::str_dump("Map after aligning its parameters with the context parameters: ",
                                     isl_map_to_str(map)));

        isl_id *missing_id1 = NULL;
        if (isl_map_has_tuple_id(map, isl_dim_in) == isl_bool_true)
        {
            missing_id1 = isl_map_get_tuple_id(map, isl_dim_in);
        }
        else
        {
            std::string name = isl_map_get_tuple_name(map, isl_dim_in);
            assert(name.size() > 0);
            missing_id1 = isl_id_alloc(this->get_ctx(), name.c_str(), NULL);
        }

        int nb_dims = isl_map_dim(map, isl_dim_in);
        context = isl_set_add_dims(context, isl_dim_set, nb_dims);
        DEBUG(10, tiramisu::str_dump("Context after adding dimensions to make it have the same number of dimensions as missing: ",
                                     isl_set_to_str (context)));
        context = isl_set_set_tuple_id(context, isl_id_copy(missing_id1));
        DEBUG(10, tiramisu::str_dump("Context after setting its tuple ID to be equal to the tuple ID of missing: ",
                                     isl_set_to_str (context)));
        map = isl_map_intersect_domain(map, isl_set_copy(context));
        DEBUG(10, tiramisu::str_dump("Map after intersecting with the program context: ",
                                     isl_map_to_str(map)));
    }

    DEBUG_INDENT(-4);

    return map;
}

/**
 * Assuming the set missing is the set of missing computations that will be
 * duplicated. The duplicated computations may needed to be shifted so that
 * they are executed with the original computation rather than being executed
 * after the original computation.
 * This function figures out the shift degree for each dimension of the missing
 * set.
 *
 * - For each dimension d in [0 to L]:
 *      * Project all the dimensions of the missing set except the dimension d.
 *      * The shift factor is obtained as follows:
 *              For the remaining the negative of the constant value of that dimension.
 */
std::vector<int> get_shift_degrees(isl_set *missing, int L)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::vector<int> shifts;

    DEBUG(3, tiramisu::str_dump("Getting the shift degrees for the missing set."));
    DEBUG(3, tiramisu::str_dump("The missing set is: ", isl_set_to_str(missing)));
    DEBUG(3, tiramisu::str_dump("Get the shift degrees up to the loop level : " + std::to_string(L)));

    for (int i = 0; i <= L; i++)
    {
        isl_set *m = isl_set_copy(missing);
        int dim = loop_level_into_dynamic_dimension(i);
        int max_dim = loop_level_into_dynamic_dimension(L);
        DEBUG(3, tiramisu::str_dump("The current dynamic dimension is: " + std::to_string(dim)));

        DEBUG(3, tiramisu::str_dump("Projecting out all the dimensions of the set except the dimension " +
                                    std::to_string(dim)));

        if (dim != 0)
        {
            m = isl_set_project_out(m, isl_dim_set, 0, dim);
            DEBUG(10, tiramisu::str_dump("Projecting " + std::to_string(dim) +
                                         " dimensions starting from dimension 0."));
        }

        DEBUG(10, tiramisu::str_dump("After projection: ", isl_set_to_str(m)));

        if (dim != max_dim)
        {
            int last_dim = isl_set_dim(m, isl_dim_set);
            DEBUG(10, tiramisu::str_dump("Projecting " + std::to_string(last_dim - 1) +
                                         " dimensions starting from dimension 1."));
            m = isl_set_project_out(m, isl_dim_set, 1, last_dim - 1);
        }

        DEBUG(3, tiramisu::str_dump("After projection: ", isl_set_to_str(m)));

        /**
         * TODO: We assume that the set after projection is of the form
         * [T0]->{[i0]: i0 = T0 + 1}
         * which is in general the case, but we need to check that this
         * is the case. If it is not the case, the computed shifts are wrong.
         * i.e., check that we do not have any other dimension or parameter is
         * involved in the constraint. The constraint should have the form
         * dynamic_dimension = fixed_dimension + constant
         * where tile_dimension is a fixed dimension and where constant is
         * a literal constant not a symbolic constant. This constant will
         * become the shift degree.
         */
        int c = (-1) * isl_set_get_const_dim(isl_set_copy(m), 0);

        shifts.push_back(c);

        DEBUG(3, tiramisu::str_dump("The constant value of the remaining dimension is: " + std::to_string(
                                        c)));
    }

    if (ENABLE_DEBUG && DEBUG_LEVEL >= 3)
    {
        DEBUG_NO_NEWLINE(3, tiramisu::str_dump("Shift degrees are: "));
        for (auto c : shifts)
        {
            tiramisu::str_dump(std::to_string(c) + " ");
        }
        tiramisu::str_dump("\n");
    }

    DEBUG_INDENT(-4);

    return shifts;
}

/**
 * Compute the needed area.
 */
std::vector<isl_set *> computation::compute_needed_and_produced(computation &consumer, int L,
        std::vector<std::string> &param_names)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::vector<isl_set *> needed_and_produced;

    // Get the consumer domain and schedule and the producer domain and schedule
    isl_set *consumer_domain = isl_set_copy(consumer.get_iteration_domain());
    isl_map *consumer_sched = isl_map_copy(consumer.get_schedule());
    isl_set *producer_domain = isl_set_copy(this->get_iteration_domain());
    isl_map *producer_sched = isl_map_copy(this->get_schedule());

    // Compute the access relation of the consumer computation.
    std::vector<isl_map *> accesses_vector;
    generator::get_rhs_accesses(consumer.get_function(), &consumer, accesses_vector, false);
    assert(accesses_vector.size() > 0);

    DEBUG(3, tiramisu::str_dump("Computed RHS accesses:"));
    for (auto acc : accesses_vector)
    {
        DEBUG(3, tiramisu::str_dump(isl_map_to_str(acc)));
    }

    DEBUG(3, tiramisu::str_dump("Vector of accesses computed."));

    // Create a union map of the accesses to the producer.
    isl_map *consumer_accesses = NULL;

    DEBUG(10, tiramisu::str_dump("Computing a union map of accesses to the producer."));

    for (const auto a : accesses_vector)
    {
        std::string range_name = isl_map_get_tuple_name(isl_map_copy(a), isl_dim_out);

        if (range_name == this->get_name())
        {
	    if (consumer_accesses == NULL)
		consumer_accesses = isl_map_copy(a);
	    else
	    {
		DEBUG(10, tiramisu::str_dump("consumer_accesses: ", isl_map_to_str(consumer_accesses)));
		DEBUG(10, tiramisu::str_dump("access: ", isl_map_to_str(a)));

		consumer_accesses = isl_map_union(isl_map_copy(a), consumer_accesses);
	    }
        }
    }

    DEBUG(10, tiramisu::str_dump("Union computed."));

    DEBUG(10, tiramisu::str_dump("Intersecting the range and the domain of the following consumer_accesses: ", isl_map_to_str(consumer_accesses)));
    DEBUG(10, tiramisu::str_dump("with the following iteration domain: ", isl_set_to_str(this->get_iteration_domain())));

    consumer_accesses = isl_map_intersect_range(consumer_accesses,
                        isl_set_copy(this->get_iteration_domain()));
    consumer_accesses = isl_map_intersect_domain(consumer_accesses,
                        isl_set_copy(consumer.get_iteration_domain()));
    consumer_accesses = this->simplify(consumer_accesses);

    DEBUG(3, tiramisu::str_dump("Accesses after keeping only those that have the producer in the range: "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(consumer_accesses)));

    // Simplify
    consumer_domain = this->simplify(consumer_domain);
    consumer_sched = this->simplify(consumer_sched);
    producer_sched = this->simplify(producer_sched);
    producer_domain = this->simplify(producer_domain);

    // Transform, into time-processor, the consumer domain and schedule and the producer domain and schedule and the access relation
    consumer_domain = isl_set_apply(consumer_domain, isl_map_copy(consumer_sched));
    assert(consumer_domain != NULL);
    producer_domain = isl_set_apply(producer_domain, isl_map_copy(producer_sched));
    assert(producer_domain != NULL);

    // Transform the consumer accesses to the time-space domain.
    // For each access of the consumer:
    //	    - Apply the schedule of the consumer on the domain of the access,
    //	    - Get the producer (range) involved in that access,
    //	    - Get the schedule of that producer,
    //	    - Apply that schedule on the range of the access,
    //	    - Add the resulting schedule to the union representing the result.
    {
	DEBUG(3, tiramisu::str_dump("Applying consumer_sched on the domain of consumer_accesses."));
	DEBUG(3, tiramisu::str_dump("consumer_sched: ", isl_map_to_str(consumer_sched)));
	DEBUG(3, tiramisu::str_dump("consumer_accesses: ", isl_map_to_str(consumer_accesses)));

	consumer_accesses = isl_map_apply_domain(isl_map_copy(consumer_accesses),
			    isl_map_copy(consumer_sched));
	assert(consumer_accesses != NULL);

	DEBUG(3, tiramisu::str_dump("Applying it on the range."));

	consumer_accesses = isl_map_apply_range(isl_map_copy(consumer_accesses),
	                                        isl_map_copy(producer_sched));
	assert(consumer_accesses != NULL);

	DEBUG(3, tiramisu::str_dump("")); DEBUG(3, tiramisu::str_dump(""));
	DEBUG(3, tiramisu::str_dump("Consumer domain (in time-processor): ",
				    isl_set_to_str(consumer_domain)));
	DEBUG(3, tiramisu::str_dump("Consumer accesses (in time-processor): ",
				    isl_map_to_str(consumer_accesses)));
	DEBUG(3, tiramisu::str_dump("Producer domain (in time-processor): ",
				    isl_set_to_str(producer_domain)));
    }

    // Add parameter dimensions and equate the dimensions on the left of dim to these parameters
    if (L + 1 > 0)
    {
        int pos_last_param0 = isl_set_dim(consumer_domain, isl_dim_param);
        int pos_last_param1 = isl_set_dim(producer_domain, isl_dim_param);
        consumer_domain = isl_set_add_dims(consumer_domain, isl_dim_param, L + 1);
        producer_domain = isl_set_add_dims(producer_domain, isl_dim_param, L + 1);

        // Set the names of the new parameters
        for (int i = 0; i <= L; i++)
        {
            std::string new_param = generate_new_variable_name();
            consumer_domain = isl_set_set_dim_name(consumer_domain, isl_dim_param, pos_last_param0 + i,
                                                   new_param.c_str());
            producer_domain = isl_set_set_dim_name(producer_domain, isl_dim_param, pos_last_param1 + i,
                                                   new_param.c_str());
            param_names.push_back(new_param);
        }

        isl_space *sp = isl_set_get_space(isl_set_copy(consumer_domain));
        isl_local_space *lsp = isl_local_space_from_space(isl_space_copy(sp));

        isl_space *sp2 = isl_set_get_space(isl_set_copy(producer_domain));
        isl_local_space *lsp2 = isl_local_space_from_space(isl_space_copy(sp2));

        for (int i = 0; i <= L; i++)
        {
            // Assuming that i is the dynamic dimension and T is the parameter.
            // We want to create the following constraint: i - T = 0
            int pos = loop_level_into_dynamic_dimension(i);
            isl_constraint *cst = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_set, pos, 1);
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_param, pos_last_param0 + i, -1);
            consumer_domain = isl_set_add_constraint(consumer_domain, cst);

            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp2));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_set, pos, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_param, pos_last_param1 + i, -1);
            producer_domain =  isl_set_add_constraint(producer_domain, cst2);
        }
    }
    DEBUG(3, tiramisu::str_dump("Consumer domain after fixing left dimensions to parameters: ",
                                isl_set_to_str(consumer_domain)));
    DEBUG(3, tiramisu::str_dump("Producer domain after fixing left dimensions to parameters: ",
                                isl_set_to_str(producer_domain)));


    // Compute needed = consumer_access(consumer_domain)
    isl_set *needed = isl_set_apply(isl_set_copy(consumer_domain), isl_map_copy(consumer_accesses));
    needed = this->simplify(needed);
    DEBUG(3, tiramisu::str_dump("Needed in time-processor = consumer_access(consumer_domain) in time-processor: ",
                                isl_set_to_str(needed)));

    needed_and_produced.push_back(needed);
    needed_and_produced.push_back(producer_domain);

    DEBUG_INDENT(-4);

    return needed_and_produced;
}


/**
 * - Get the access function of the consumer (access to computations).
 * - Apply the schedule on the iteration domain and access functions.
 * - Keep only the access function to the producer.
 * - Compute the iteration space of the consumer with all dimensions after L projected out.
 * - Project out the dimensions after L in the access function.
 * - Compute the image of the iteration space with the access function.
 *   //This is called the "needed".
 *
 * - Project out the dimensions that are after L in the iteration domain of the producer.
 *   // This is called the "produced".
 *
 * -  missing = needed - produced.
 *
 * - Add universal dimensions to the missing set.
 *
 * - Use the missing set as an argument to create the redundant computation.
 *
 * - How to shift:
 *      max(needed) - max(produced) at the level L. The result should be an integer.
 *
 * - Order the redundant computation after the original at level L.
 * - Order the consumer after the redundant at level L.
 */
// TODO: Test the case when \p consumer does not consume this computation.
void computation::compute_at(computation &consumer, int L)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L > 0);

    int dim = loop_level_into_static_dimension(L);

    assert(this->get_schedule() != NULL);
    assert(dim < (signed int) isl_map_dim(isl_map_copy(this->get_schedule()), isl_dim_out));
    assert(dim >= computation::root_dimension);

    this->get_function()->align_schedules();

    DEBUG(3, tiramisu::str_dump("Setting the schedule of the producer ");
          tiramisu::str_dump(this->get_name());
          tiramisu::str_dump(" to be computed at the loop nest of the consumer ");
          tiramisu::str_dump(consumer.get_name());
          tiramisu::str_dump(" at dimension ");
          tiramisu::str_dump(std::to_string(dim)));
    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(this->get_schedule())));

    // Compute needed
    std::vector<std::string> param_names;
    std::vector<isl_set *> needed_and_produced = this->compute_needed_and_produced(consumer, L,
            param_names);
    isl_set *needed = needed_and_produced[0];
    isl_set *producer_domain = needed_and_produced[1];

    // Compute missing = needed - producer
    // First, rename the needed to have the same space name as produced
    needed = isl_set_set_tuple_name(needed, isl_set_get_tuple_name(isl_set_copy(producer_domain)));

    /*
     * The isl_set_subtract function is not well documented. Here is a test that indicates what is does exactly.
     * S1: { S[i, j] : i >= 0 and i <= 100 and j >= 0 and j <= 100 }
     * S2: { S[i, j] : i >= 0 and i <= 50 and j >= 0 and j <= 50 }
     * isl_set_subtract(S2, S1): { S[i, j] : 1 = 0 }
     * isl_set_subtract(S1, S2): { S[i, j] : (i >= 51 and i <= 100 and j >= 0 and j <= 100) or (i >= 0 and i <= 50 and j >= 51 and j <= 100) }
     *
     * So isl_set_subtract(S1, S2) = S1 - S2.
     */
    isl_set *missing = isl_set_subtract(isl_set_copy(needed), isl_set_copy(producer_domain));
    missing = this->simplify(missing);
    DEBUG(3, tiramisu::str_dump("Missing = needed - producer = ", isl_set_to_str(missing)));
    DEBUG(3, tiramisu::str_dump("")); DEBUG(3, tiramisu::str_dump(""));
    isl_set *original_missing = isl_set_copy(missing);

    if (!isl_set_is_empty(missing))
    {
        std::vector<int> shift_degrees = get_shift_degrees(isl_set_copy(missing), L);

        // Now replace the parameters by existential variables and remove them
        if (L + 1 > 0)
        {
            int pos_last_dim = isl_set_dim(missing, isl_dim_set);
            std::string space_name = isl_set_get_tuple_name(missing);
            missing = isl_set_add_dims(missing, isl_dim_set, L + 1);
            missing = isl_set_set_tuple_name(missing, space_name.c_str());

            // Set the names of the new dimensions.
            for (int i = 0; i <= L; i++)
            {
                missing = isl_set_set_dim_name(missing, isl_dim_set, pos_last_dim + i,
                                               ("p" + param_names[i]).c_str());
            }

            /* Go through all the constraints of the set "missing" and replace them with new constraints.
             * In the new constraints, each coefficient of a param is replaced by a coefficient to the new
             * dynamic variables. Later, these dynamic variables are projected out to create existential
             * variables.
             *
             * For each basic set in a set
             *      For each constraint in a basic set
             *          For each parameter variable created previously
             *              If the constraint involves that parameter
             *                  Read the coefficient of the parameter.
             *                  Set the coefficient of the corresponding variable into that coefficient
             *                  and set the coefficient of the parameter to 0.
             * Project out the dynamic variables.  The parameters are kept but are not used at all in the
             * constraints of "missing".
             */
            isl_set *new_missing = isl_set_universe(isl_space_copy(isl_set_get_space(isl_set_copy(missing))));
            isl_basic_set_list *bset_list = isl_set_get_basic_set_list(isl_set_copy(missing));
            for (int i = 0; i < isl_set_n_basic_set(missing); i++)
            {
                isl_basic_set *bset = isl_basic_set_list_get_basic_set(isl_basic_set_list_copy(bset_list), i);
                isl_basic_set *new_bset = isl_basic_set_universe(isl_space_copy(isl_basic_set_get_space(
                                              isl_basic_set_copy(bset))));
                isl_constraint_list *cst_list = isl_basic_set_get_constraint_list(bset);
                isl_space *sp = isl_basic_set_get_space(bset);
                DEBUG(10, tiramisu::str_dump("Retrieving the constraints of the bset:",
                                             isl_set_to_str(isl_set_from_basic_set(isl_basic_set_copy(bset)))));
                DEBUG(10, tiramisu::str_dump("Number of constraints: " + std::to_string(
                                                 isl_constraint_list_n_constraint(cst_list))));
                DEBUG(10, tiramisu::str_dump("List of constraints: "); isl_constraint_list_dump(cst_list));

                for (int j = 0; j < isl_constraint_list_n_constraint(cst_list); j++)
                {
                    DEBUG(10, tiramisu::str_dump("Checking the constraint number " + std::to_string(j)));
                    isl_constraint *cst = isl_constraint_list_get_constraint(cst_list, j);
                    DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Constraint: "); isl_constraint_dump(cst));
                    for (auto const p : param_names)
                    {
                        int pos = isl_space_find_dim_by_name(sp, isl_dim_param, p.c_str());
                        if (isl_constraint_involves_dims(cst, isl_dim_param, pos, 1))
                        {
                            DEBUG(10, tiramisu::str_dump("Does the constraint involve the parameter " + p + "? Yes."));
                            DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Modifying the constraint. The original constraint:");
                                             isl_constraint_dump(cst));
                            isl_val *coeff = isl_constraint_get_coefficient_val(cst, isl_dim_param, pos);
                            cst = isl_constraint_set_coefficient_si(cst, isl_dim_param, pos, 0);
                            int pos2 = isl_space_find_dim_by_name(sp, isl_dim_set, ("p" + p).c_str());
                            cst = isl_constraint_set_coefficient_val(cst, isl_dim_set, pos2, isl_val_copy(coeff));
                            DEBUG_NO_NEWLINE(10, tiramisu::str_dump("The new constraint:"); isl_constraint_dump(cst));
                        }
                        else
                        {
                            DEBUG(10, tiramisu::str_dump("Does the constraint involve the parameter " + p + "? No."));
                        }
                    }
                    DEBUG(10, tiramisu::str_dump(""));

                    new_bset = isl_basic_set_add_constraint(new_bset, isl_constraint_copy(cst));
                }

                DEBUG(10, tiramisu::str_dump("The basic set after modifying the constraints:");
                      isl_basic_set_dump(new_bset));

                // In the first time, restrict the universal new_missing with the new bset,
                // in the next times compute the union of the bset with new_missing.
                if (i == 0)
                {
                    new_missing = isl_set_intersect(new_missing, isl_set_from_basic_set(new_bset));
                }
                else
                {
                    new_missing = isl_set_union(new_missing, isl_set_from_basic_set(new_bset));
                }

                DEBUG(10, tiramisu::str_dump("The new value of missing (after intersecting with the new bset):");
                      isl_set_dump(new_missing));

            }
            missing = new_missing;

            // Project out the set dimensions to make them existential variables
            missing = isl_set_project_out(missing, isl_dim_set, pos_last_dim, L + 1);
            int pos_first_param = isl_space_find_dim_by_name(isl_set_get_space(missing), isl_dim_param,
                                  param_names[0].c_str());
            missing = isl_set_project_out(missing, isl_dim_param, pos_first_param, L + 1);
            missing = isl_set_set_tuple_name(missing, space_name.c_str());

            DEBUG(3, tiramisu::str_dump("Missing before replacing the parameters with existential variables: ",
                                        isl_set_to_str(original_missing)));
            DEBUG(3, tiramisu::str_dump("Missing after replacing the parameters with existential variables: ",
                                        isl_set_to_str(missing)));
            DEBUG(3, tiramisu::str_dump(""));
        }
        // Duplicate the producer using the missing set which is in the time-processor domain.
        tiramisu::computation *original_computation = this;
        tiramisu::computation *duplicated_computation = this->duplicate("", isl_set_to_str(missing));
	this->updates.push_back(duplicated_computation);
        DEBUG(3, tiramisu::str_dump("Producer duplicated. Dumping the schedule of the original computation."));
        original_computation->dump_schedule();
        DEBUG(3, tiramisu::str_dump("Dumping the schedule of the duplicate computation."));
        duplicated_computation->dump_schedule();

        DEBUG(3, tiramisu::str_dump("Now setting the duplicate with regard to the other computations."));
        original_computation->after((*duplicated_computation), L);
        consumer.after((*original_computation), L);

        // Computing the shift degrees.
        for (int i = 0; i <= L; i++)
            if (shift_degrees[i] != 0)
            {
                DEBUG(3, tiramisu::str_dump("Now shifting the duplicate by " + std::to_string(
                                                shift_degrees[i]) + " at loop level " + std::to_string(i)));
                duplicated_computation->shift(i, shift_degrees[i]);
            }
    }
    else
    {
        tiramisu::computation *original_computation = this;
        consumer.after((*original_computation), L);
    }
    DEBUG(3, tiramisu::str_dump("Dumping the schedule of the producer and consumer."));
    this->dump_schedule();
    consumer.dump_schedule();

    DEBUG_INDENT(-4);
}

/**
  * Wrapper around compute_at(computation &consumer, int L).
  */
void computation::compute_at(computation &consumer, tiramisu::var L_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L_var.get_name().size() > 0);

    std::vector<int> dimensions = consumer.get_loop_level_numbers_from_dimension_names({L_var.get_name()});
    assert(dimensions.size() == 1);

    int L = dimensions[0];

    this->compute_at(consumer, L);

    DEBUG_INDENT(-4);
}

/**
 * Return true if \p cst is a simple constraint, i.e., it satisfies the
 * following conditions:
 *  - It involves only the dimension \p dim and does not involve any
 *    other dimension,
 *  - It has 1 as a coefficient for \p dim
 */
bool isl_constraint_is_simple(isl_constraint *cst, int dim)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    bool simple = true;

    isl_space *space = isl_constraint_get_space(cst);
    for (int i = 0; i < isl_space_dim(space, isl_dim_set); i++)
        if (i != dim)
            if (isl_constraint_involves_dims(cst, isl_dim_set, i, 1))
            {
                DEBUG(10, tiramisu::str_dump("Constraint involves multiple dimensions"));
                simple = false;
            }

    isl_val *coeff = isl_constraint_get_coefficient_val(cst, isl_dim_set, dim);
    if ((isl_val_is_negone(coeff) == isl_bool_false) && (isl_val_is_one(coeff) == isl_bool_false))
    {
        DEBUG(10, tiramisu::str_dump("Coefficient of the dimension is not one/negative(one).");
              isl_val_dump(coeff));
        simple = false;
    }

    DEBUG_INDENT(-4);

    return simple;
}


/**
 * Extract a tiramisu expression that represents the bound on the dimension
 * \p dim in the constraint \p cst.
 *
 * If \p upper is true, then the bound is an upper bound, otherwise the bound
 * is a lower bound.
 */
tiramisu::expr extract_tiramisu_expr_from_cst(isl_constraint *cst, int dim, bool upper)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(cst != NULL);

    isl_space *space = isl_constraint_get_space(cst);
    tiramisu::expr e = tiramisu::expr();

    DEBUG(10, tiramisu::str_dump("Computing the expression that correspond to the following constraint at dimension "
                                 + std::to_string(dim) + " : "));
    DEBUG(10, isl_constraint_dump(cst));

    // Add the parameter to the expression
    for (int i = 0; i < isl_space_dim(space, isl_dim_param); i++)
    {
        isl_val *coeff = isl_constraint_get_coefficient_val(cst, isl_dim_param, i);
        if (isl_val_is_zero(coeff) == isl_bool_false)
        {
            const char *name = isl_space_get_dim_name(space, isl_dim_param, i);
            tiramisu::expr param = tiramisu::var(global::get_loop_iterator_data_type(), std::string(name));
            if (isl_val_is_one(coeff) == isl_bool_false)
            {
                long c = isl_val_get_num_si(coeff);

                // For lower bounds, inverse the sign.
                if (upper == false)
                {
                    c = -1 * c;
                }

                param = tiramisu::expr(o_mul,
                                       tiramisu::expr(o_cast, tiramisu::global::get_loop_iterator_data_type(),
                                                      tiramisu::expr((int32_t) c)), param);
            }

            if (e.is_defined() == false)
            {
                e = param;
            }
            else
            {
                e = tiramisu::expr(o_add, e, param);
            }
        }
    }

    isl_val *ct = isl_constraint_get_constant_val(cst);
    if ((isl_val_is_zero(ct) == isl_bool_false) || (e.is_defined() == false))
    {
        long v = isl_val_get_num_si(ct);

        // For lower bounds, inverse the sign.
        if (upper == false)
        {
            v = -1 * v;
        }

        tiramisu::expr c = tiramisu::expr(o_cast, global::get_loop_iterator_data_type(), tiramisu::expr((int32_t) v));

        if (e.is_defined() == false)
        {
            e = c;
        }
        else
        {
            e = tiramisu::expr(o_add, e, c);
        }
    }

    DEBUG(10, tiramisu::str_dump("The expression that correspond to the expression is : ");
          e.dump(false));
    DEBUG_INDENT(-4);

    return e;
}

int compute_recursively_max_AST_depth(isl_ast_node *node)
{
    assert(node != NULL);

    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    int result = -1;

    DEBUG(10, tiramisu::str_dump("Computing maximal AST depth from the following ISL AST node "));
    DEBUG(10, tiramisu::str_dump(std::string(isl_ast_node_to_C_str(node))));

    if (isl_ast_node_get_type(node) == isl_ast_node_block)
    {
        DEBUG(10, tiramisu::str_dump("Computing maximal depth from a block."));

        isl_ast_node_list *list = isl_ast_node_block_get_children(node);
        isl_ast_node *child = isl_ast_node_list_get_ast_node(list, 0);
	result = compute_recursively_max_AST_depth(child);

        for (int i = 1; i < isl_ast_node_list_n_ast_node(list); i++)
        {
            child = isl_ast_node_list_get_ast_node(list, i);
	    result = std::max(result, compute_recursively_max_AST_depth(child));
        }
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_for)
    {
        DEBUG(10, tiramisu::str_dump("Computing maximal depth from a for loop."));
        isl_ast_node *body = isl_ast_node_for_get_body(node);
        result = compute_recursively_max_AST_depth(body) + 1;
        isl_ast_node_free(body);
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_user)
    {
        DEBUG(10, tiramisu::str_dump("Reached a user node."));
	return 1;
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_if)
    {
        DEBUG(10, tiramisu::str_dump("Computing maximal depth from an if conditional."));

	result = compute_recursively_max_AST_depth(isl_ast_node_if_get_then(node));

	if (isl_ast_node_if_has_else(node))
	    result = std::max(result, compute_recursively_max_AST_depth(isl_ast_node_if_get_else(node)));
    }
    else
    {
	tiramisu::error("Found an unsupported ISL AST node while computing the maximal AST depth.", true);
    }

    DEBUG(3, tiramisu::str_dump("Current depth = " + std::to_string(result)));
    DEBUG_INDENT(-4);

    return result;
}

/**
  * Traverse recursively the ISL AST tree
  *
  * \p node represents the root of the tree to be traversed.
  *
  * \p dim is the dimension of the loop from which the bounds have to be
  * extracted.
  *
  * \p upper is a boolean that should be set to true to extract
  * the upper bound and false to extract the lower bound.
  */
tiramisu::expr utility::extract_bound_expression(isl_ast_node *node, int dim, bool upper)
{
    assert(node != NULL);
    assert(dim >= 0);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::expr result;

    DEBUG(3, tiramisu::str_dump("Extracting bounds from a loop at depth = " + std::to_string(dim)));
    DEBUG(3, tiramisu::str_dump("Extracting bounds from the following ISL AST node "));
    DEBUG(3, tiramisu::str_dump(std::string(isl_ast_node_to_C_str(node))));

    if (isl_ast_node_get_type(node) == isl_ast_node_block)
	tiramisu::error("Currently Tiramisu does not support extracting bounds from blocks.", true);
    else if (isl_ast_node_get_type(node) == isl_ast_node_for)
    {
        DEBUG(3, tiramisu::str_dump("Extracting bounds from a for loop."));
        isl_ast_expr *init_bound = isl_ast_node_for_get_init(node);
        isl_ast_expr *upper_bound = isl_ast_node_for_get_cond(node);
        DEBUG(3, tiramisu::str_dump("Lower bound at this level is: " + std::string(isl_ast_expr_to_C_str(init_bound))));
        DEBUG(3, tiramisu::str_dump("Upper bound at this level is: " + std::string(isl_ast_expr_to_C_str(upper_bound))));

	if (dim == 0)
	{
            if (upper)
	    {
                isl_ast_expr *cond = isl_ast_node_for_get_cond(node);

		/**
		  * If we have an expression
		  *  i < N
		  * or an expression
		  *  i <= N - 1
		  *
		  * In both cases, the returned bound should be (N-1).
		  */
                if (isl_ast_expr_get_op_type(cond) == isl_ast_op_lt)
                {
                    // Create an expression of "1".
                    isl_val *one = isl_val_one(isl_ast_node_get_ctx(node));
                    // Add 1 to the ISL ast upper bound to transform it into a strinct bound.
                    result = tiramisu_expr_from_isl_ast_expr(isl_ast_expr_sub(isl_ast_expr_get_op_arg(cond, 1),
									      isl_ast_expr_from_val(one)));
                }
                else if (isl_ast_expr_get_op_type(cond) == isl_ast_op_le)
                {
                    result = tiramisu_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(cond, 1));
                }
	   }
	   else
	   {
                isl_ast_expr *init = isl_ast_node_for_get_init(node);
		result = tiramisu_expr_from_isl_ast_expr(init);
	   }
	}
	else
	{
            isl_ast_node *body = isl_ast_node_for_get_body(node);
	    result = utility::extract_bound_expression(body, dim-1, upper);
            isl_ast_node_free(body);
	}

        assert(result.is_defined());
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_user)
	tiramisu::error("Cannot extract bounds from a isl_ast_user node.", true);
    else if (isl_ast_node_get_type(node) == isl_ast_node_if)
    {
        DEBUG(3, tiramisu::str_dump("If conditional."));

	// tiramisu::expr cond_bound = tiramisu_expr_from_isl_ast_expr(isl_ast_node_if_get_cond(node));
	tiramisu::expr then_bound = utility::extract_bound_expression(isl_ast_node_if_get_then(node), dim, upper);

	tiramisu::expr else_bound;
	if (isl_ast_node_if_has_else(node))
	{
	    // else_bound = utility::extract_bound_expression(isl_ast_node_if_get_else(node), dim, upper);
	    // result = tiramisu::expr(tiramisu::o_s, cond_bound, then_bound, else_bound);
	    tiramisu::error("If Then Else is unsupported in bound extraction.", true);
	}
	else
	    result = then_bound; //tiramisu::expr(tiramisu::o_cond, cond_bound, then_bound);
    }

    DEBUG(3, tiramisu::str_dump("Extracted bound:"); result.dump(false));
    DEBUG_INDENT(-4);

    return result;
}

int computation::compute_maximal_AST_depth()
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    this->name_unnamed_time_space_dimensions();
    this->gen_time_space_domain();
    isl_set *set = this->get_trimmed_time_processor_domain();
    assert(set != NULL);

    DEBUG(10, tiramisu::str_dump(std::string("Getting the ") +
                                 " maximal AST depth of the set ",
                                 isl_set_to_str(set)));

    isl_ast_build *ast_build;
    isl_ctx *ctx = isl_set_get_ctx(set);
    ast_build = isl_ast_build_alloc(ctx);

    // Create identity map for set.
    isl_space *sp = isl_set_get_space(set);
    isl_map *sched = isl_map_identity(isl_space_copy(isl_space_map_from_set(sp)));
    sched = isl_map_set_tuple_name(sched, isl_dim_out, "");

    // Generate the AST.
    DEBUG(10, tiramisu::str_dump("Setting ISL AST generator options."));
    isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
    isl_options_get_ast_build_exploit_nested_bounds(ctx);
    isl_options_set_ast_build_group_coscheduled(ctx, 1);
    isl_options_set_ast_build_allow_else(ctx, 1);
    isl_options_set_ast_build_detect_min_max(ctx, 1);

    // Intersect the iteration domain with the domain of the schedule.
    DEBUG(10, tiramisu::str_dump("Generating time-space domain."));
    isl_map *map = isl_map_intersect_domain(isl_map_copy(sched), isl_set_copy(set));

    // Set iterator names
    DEBUG(10, tiramisu::str_dump("Setting the iterator names."));
    int length = isl_map_dim(map, isl_dim_out);
    isl_id_list *iterators = isl_id_list_alloc(ctx, length);

    for (int i = 0; i < length; i++)
    {
        std::string name;
        if (isl_set_has_dim_name(set, isl_dim_set, i) == true)
            name = isl_set_get_dim_name(set, isl_dim_set, i);
        else
            name = generate_new_variable_name();
	isl_id *id = isl_id_alloc(ctx, name.c_str(), NULL);
        iterators = isl_id_list_add(iterators, id);
    }

    ast_build = isl_ast_build_set_iterators(ast_build, iterators);

    isl_ast_node *node = isl_ast_build_node_from_schedule_map(ast_build, isl_union_map_from_map(map));
    int depth = compute_recursively_max_AST_depth(node);
    isl_ast_build_free(ast_build);

    DEBUG(10, tiramisu::str_dump("The maximal AST depth is : " + std::to_string(depth)));
    DEBUG_INDENT(-4);

    return depth;
}

/**
 * - Generate code:
 *	- Generate time-processor domain.
 *	- Generate an ISL AST.
 * - Traverse the tree until the level \p dim.
 * - Extract the bounds of that level.
 * - During the traversal, assert that the loop is fully nested.
 *
 */
tiramisu::expr utility::get_bound(isl_set *set, int dim, int upper)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(set != NULL);
    assert(dim >= 0);
    assert(dim < isl_space_dim(isl_set_get_space(set), isl_dim_set));
    assert(isl_set_is_empty(set) == isl_bool_false);

    DEBUG(10, tiramisu::str_dump(std::string("Getting the ") + (upper ? "upper" : "lower") +
                                 " bound on the dimension " +
                                 std::to_string(dim) + " of the set ",
                                 isl_set_to_str(set)));

    tiramisu::expr e = tiramisu::expr();
    isl_ast_build *ast_build;
    isl_ctx *ctx = isl_set_get_ctx(set);
    ast_build = isl_ast_build_alloc(ctx);

    // Create identity map for set.
    isl_space *sp = isl_set_get_space(set);
    isl_map *sched = isl_map_identity(isl_space_copy(isl_space_map_from_set(sp)));
    sched = isl_map_set_tuple_name(sched, isl_dim_out, "");

    // Generate the AST.
    DEBUG(3, tiramisu::str_dump("Setting ISL AST generator options."));
    isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
    isl_options_get_ast_build_exploit_nested_bounds(ctx);
    isl_options_set_ast_build_group_coscheduled(ctx, 1);
    isl_options_set_ast_build_allow_else(ctx, 1);
    isl_options_set_ast_build_detect_min_max(ctx, 1);

    // Computing the polyhedral hull of the input set.
    //DEBUG(3, tiramisu::str_dump("Computing the polyhedral hull of the input set."));
    //set = isl_set_from_basic_set(isl_set_affine_hull(isl_set_copy(set)));
    //DEBUG(3, tiramisu::str_dump("The polyhedral hull is: ", isl_set_to_str(set)));

    // Intersect the iteration domain with the domain of the schedule.
    DEBUG(3, tiramisu::str_dump("Generating time-space domain."));
    isl_map *map =
        isl_map_intersect_domain(
            isl_map_copy(sched),
            isl_set_copy(set));

    // Set iterator names
    DEBUG(3, tiramisu::str_dump("Setting the iterator names."));
    int length = isl_map_dim(map, isl_dim_out);
    isl_id_list *iterators = isl_id_list_alloc(ctx, length);

    for (int i = 0; i < length; i++)
    {
        std::string name;
        if (isl_set_has_dim_name(set, isl_dim_set, i) == true)
            name = isl_set_get_dim_name(set, isl_dim_set, i);
        else
            name = generate_new_variable_name();
	isl_id *id = isl_id_alloc(ctx, name.c_str(), NULL);
        iterators = isl_id_list_add(iterators, id);
    }

    ast_build = isl_ast_build_set_iterators(ast_build, iterators);

    isl_ast_node *node = isl_ast_build_node_from_schedule_map(ast_build, isl_union_map_from_map(map));
    e = utility::extract_bound_expression(node, dim, upper);
    isl_ast_build_free(ast_build);

    assert(e.is_defined() && "The computed bound expression is undefined.");
    DEBUG(10, tiramisu::str_dump(std::string("The ") + (upper ? "upper" : "lower") + " bound is : "); e.dump(false));
    DEBUG_INDENT(-4);

    return e;
}

bool computation::separateAndSplit(tiramisu::var L0, int sizeX)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::var L0_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L0_inner = tiramisu::var(generate_new_variable_name());

    bool split_happened = this->separateAndSplit(L0, sizeX, L0_outer, L0_inner);

    DEBUG_INDENT(-4);

    return split_happened;
}


bool computation::separateAndSplit(int L0, int v)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Applying separateAndSplit on loop level " + std::to_string(L0) + " with a split factor of " + std::to_string(v)));

    this->gen_time_space_domain();

    // Compute the depth before any scheduling.
    int original_depth = this->compute_maximal_AST_depth();

    tiramisu::expr loop_upper_bound =
        tiramisu::expr(o_cast, global::get_loop_iterator_data_type(),
                       tiramisu::utility::get_bound(this->get_trimmed_time_processor_domain(), L0, true));

    tiramisu::expr loop_lower_bound =
        tiramisu::expr(o_cast, global::get_loop_iterator_data_type(),
                       tiramisu::utility::get_bound(this->get_trimmed_time_processor_domain(), L0, false));

    tiramisu::expr loop_bound = loop_upper_bound - loop_lower_bound +
            tiramisu::expr(o_cast, global::get_loop_iterator_data_type(), tiramisu::expr((int32_t) 1));
    loop_bound = loop_bound.simplify();

    DEBUG(3, tiramisu::str_dump("Loop bound for the loop to be separated and split: "); loop_bound.dump(false));

    /*
     * Separate this computation. That is, create two identical computations
     * where we have the constraint
     *     i < v * floor(loop_bound/v)
     * in the first and
     *     i >= v * floor(loop_bound/v)
     * in the second.
     *
     * The first is called the full computation while the second is called
     * the separated computation.
     * The two computations are identical in every thing except that they have
     * two different schedules.  Their schedule restricts them to a smaller domain
     * (the full or the separated domains) and schedule one after the other.
     */
    this->separate(L0, loop_bound, v);

    /**
     * Split the full computation since the full computation will be vectorized.
     */
    this->get_update(0).split(L0, v);

    // Compute the depth after scheduling.
    int depth = this->compute_maximal_AST_depth();

    bool split_happened = false;
    if (depth == original_depth)
    {
        DEBUG(3, tiramisu::str_dump("Split happened."));

	split_happened = false;
    }
    else
    {
	 split_happened = true;
         DEBUG(3, tiramisu::str_dump("Split did not happen."));
    }

    this->get_function()->align_schedules();

    DEBUG_INDENT(-4);

    return split_happened;
}


bool computation::separateAndSplit(tiramisu::var L0_var, int v,
	    tiramisu::var L0_outer, tiramisu::var L0_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    bool split_happened = this->separateAndSplit(L0, v);

    if (split_happened == false)
    {
 	// Replace the original dimension name with the name of the outermost loop
    	this->update_names(original_loop_level_names, {L0_outer.get_name()}, L0, 1);
    }
    else
    {
	 // Replace the original dimension name with two new dimension names
    	 this->update_names(original_loop_level_names, {L0_outer.get_name(), L0_inner.get_name()}, L0, 1);
    }

    return split_happened;
}

void computation::split(tiramisu::var L0_var, int sizeX)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::var L0_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L0_inner = tiramisu::var(generate_new_variable_name());
    this->split(L0_var, sizeX, L0_outer, L0_inner);

    DEBUG_INDENT(-4);
}

void computation::split(tiramisu::var L0_var, int sizeX,
	tiramisu::var L0_outer, tiramisu::var L0_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);

    std::vector<std::string> original_loop_level_names =
	this->get_loop_level_names();

    std::vector<int> dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    this->assert_names_not_assigned({L0_outer.get_name(), L0_inner.get_name()});

    this->split(L0, sizeX);

    this->update_names(original_loop_level_names, {L0_outer.get_name(), L0_inner.get_name()}, L0, 1);

    DEBUG_INDENT(-4);
}

/**
 * Modify the schedule of this computation so that it splits the
 * loop level L0 into two new loop levels.
 * The size of the inner dimension created is sizeX.
 */
void computation::split(int L0, int sizeX)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int inDim0 = loop_level_into_dynamic_dimension(L0);

    assert(this->get_schedule() != NULL);
    assert(inDim0 >= 0);
    assert(inDim0 < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));
    assert(sizeX >= 1);

    isl_map *schedule = this->get_schedule();
    int duplicate_ID = isl_map_get_static_dim(schedule, 0);

    schedule = isl_map_copy(schedule);
    schedule = isl_map_set_tuple_id(schedule, isl_dim_out,
                                    isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL));


    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(schedule)));
    DEBUG(3, tiramisu::str_dump("Splitting dimension " + std::to_string(inDim0)
                                + " with split size " + std::to_string(sizeX)));

    std::string inDim0_str;

    std::string outDim0_str = generate_new_variable_name();
    std::string outDim1_str = generate_new_variable_name();

    int n_dims = isl_map_dim(this->get_schedule(), isl_dim_out);
    std::vector<isl_id *> dimensions;
    std::vector<std::string> dimensions_str;
    std::string map = "{";

    // -----------------------------------------------------------------
    // Preparing a map to split the duplicate computation.
    // -----------------------------------------------------------------

    map = map + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            std::string dim_str = generate_new_variable_name();
            dimensions_str.push_back(dim_str);
            map = map + dim_str;
        }
        else
        {
            std::string dim_str = generate_new_variable_name();
            dimensions_str.push_back(dim_str);
            map = map + dim_str;

            if (i == inDim0)
            {
                inDim0_str = dim_str;
            }
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] -> " + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else if (i != inDim0)
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else
        {
            map = map + outDim0_str + ", 0, " + outDim1_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim0_str.c_str(), NULL);
            isl_id *id1 = isl_id_alloc(this->get_ctx(),
                                       outDim1_str.c_str(), NULL);
            dimensions.push_back(id0);
            dimensions.push_back(id1);
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID) + " and " +
          outDim0_str + " = floor(" + inDim0_str + "/" +
          std::to_string(sizeX) + ") and " + outDim1_str + " = (" +
          inDim0_str + "%" + std::to_string(sizeX) + ")}";

    isl_map *transformation_map = isl_map_read_from_str(this->get_ctx(), map.c_str());

    for (int i = 0; i < dimensions.size(); i++)
        transformation_map = isl_map_set_dim_id(
                                 transformation_map, isl_dim_out, i, isl_id_copy(dimensions[i]));

    transformation_map = isl_map_set_tuple_id(
                             transformation_map, isl_dim_in,
                             isl_map_get_tuple_id(isl_map_copy(schedule), isl_dim_out));
    isl_id *id_range = isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL);
    transformation_map = isl_map_set_tuple_id(transformation_map, isl_dim_out, id_range);

    DEBUG(3, tiramisu::str_dump("Transformation map : ",
                                isl_map_to_str(transformation_map)));

    schedule = isl_map_apply_range(isl_map_copy(schedule), isl_map_copy(transformation_map));

    DEBUG(3, tiramisu::str_dump("Schedule after splitting: ", isl_map_to_str(schedule)));

    this->set_schedule(schedule);

    DEBUG_INDENT(-4);
}

// Methods related to the tiramisu::function class.

std::string tiramisu::function::get_gpu_thread_iterator(const std::string &comp, int lev0) const
{
    assert(!comp.empty());
    assert(lev0 >= 0);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::string res = "";

    for (const auto &pd : this->gpu_thread_dimensions)
    {
        if ((pd.first == comp) && ((std::get<0>(pd.second) == lev0) || (std::get<1>(pd.second) == lev0) ||
                                   (std::get<2>(pd.second) == lev0)))
        {
            if (lev0 == std::get<0>(pd.second))
            {
                res = ".__thread_id_z";
            }
            else if (lev0 == std::get<1>(pd.second))
            {
                res = ".__thread_id_y";
            }
            else if (lev0 == std::get<2>(pd.second))
            {
                res = ".__thread_id_x";
            }
            else
            {
                tiramisu::error("Level not mapped to GPU.", true);
            }

            std::string str = "Dimension " + std::to_string(lev0) +
                              " should be mapped to iterator " + res;
            str = str + ". It was compared against: " + std::to_string(std::get<0>(pd.second)) +
                  ", " + std::to_string(std::get<1>(pd.second)) + " and " +
                  std::to_string(std::get<2>(pd.second));
            DEBUG(3, tiramisu::str_dump(str));
        }
    }

    DEBUG_INDENT(-4);
    return res;
}

std::string tiramisu::function::get_gpu_block_iterator(const std::string &comp, int lev0) const
{
    assert(!comp.empty());
    assert(lev0 >= 0);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::string res = "";;

    for (const auto &pd : this->gpu_block_dimensions)
    {
        if ((pd.first == comp) && ((std::get<0>(pd.second) == lev0) || (std::get<1>(pd.second) == lev0) ||
                                   (std::get<2>(pd.second) == lev0)))
        {
            if (lev0 == std::get<0>(pd.second))
            {
                res = ".__block_id_z";
            }
            else if (lev0 == std::get<1>(pd.second))
            {
                res = ".__block_id_y";
            }
            else if (lev0 == std::get<2>(pd.second))
            {
                res = ".__block_id_x";
            }
            else
            {
                tiramisu::error("Level not mapped to GPU.", true);
            }

            std::string str = "Dimension " + std::to_string(lev0) +
                              " should be mapped to iterator " + res;
            str = str + ". It was compared against: " + std::to_string(std::get<0>(pd.second)) +
                  ", " + std::to_string(std::get<1>(pd.second)) + " and " +
                  std::to_string(std::get<2>(pd.second));
            DEBUG(3, tiramisu::str_dump(str));
        }
    }

    DEBUG_INDENT(-4);
    return res;
}

bool tiramisu::function::should_unroll(const std::string &comp, int lev0) const
{
    assert(!comp.empty());
    assert(lev0 >= 0);

    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    bool found = false;
    for (const auto &pd : this->unroll_dimensions)
    {
        if ((pd.first == comp) && (pd.second == lev0))
        {
            found = true;
        }
    }

    std::string str = "Dimension " + std::to_string(lev0) +
                      (found ? " should" : " should not") +
                      " be unrolled.";
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);
    return found;
}

bool tiramisu::function::should_map_to_gpu_block(const std::string &comp, int lev0) const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev0 >= 0);

    bool found = false;
    for (const auto &pd : this->gpu_block_dimensions)
    {
        if ((pd.first == comp) && ((std::get<0>(pd.second) == lev0) || (std::get<1>(pd.second) == lev0) ||
                                   (std::get<2>(pd.second) == lev0)))
        {
            found = true;
        }
    }

    std::string str = "Dimension " + std::to_string(lev0) +
                      (found ? " should" : " should not")
                      + " be mapped to GPU block.";
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);
    return found;
}

bool tiramisu::function::should_map_to_gpu_thread(const std::string &comp, int lev0) const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev0 >= 0);

    bool found = false;
    for (const auto &pd : this->gpu_thread_dimensions)
    {
        if ((pd.first == comp) && ((std::get<0>(pd.second) == lev0) || (std::get<1>(pd.second) == lev0) ||
                                   (std::get<2>(pd.second) == lev0)))
        {
            found = true;
        }
    }

    std::string str = "Dimension " + std::to_string(lev0) +
                      (found ? " should" : " should not")
                      + " be mapped to GPU thread.";
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);
    return found;
}

int tiramisu::function::get_max_identity_schedules_range_dim() const
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int max_dim = 0;
    for (const auto &comp : this->get_computations())
    {
        isl_map *sched = comp->gen_identity_schedule_for_time_space_domain();
        int m = isl_map_dim(sched, isl_dim_out);
        max_dim = std::max(max_dim, m);
    }

    DEBUG_INDENT(-4);

    return max_dim;
}

int tiramisu::function::get_max_iteration_domains_dim() const
{
    int max_dim = 0;
    for (const auto &comp : this->get_computations())
    {
        isl_set *domain = comp->get_iteration_domain();
        int m = isl_set_dim(domain, isl_dim_set);
        max_dim = std::max(max_dim, m);
    }

    return max_dim;
}

int tiramisu::function::get_max_schedules_range_dim() const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    int max_dim = 0;
    for (const auto &comp : this->get_computations())
    {
        isl_map *sched = comp->get_schedule();
        int m = isl_map_dim(sched, isl_dim_out);
        max_dim = std::max(max_dim, m);
    }

    DEBUG_INDENT(-4);

    return max_dim;
}

isl_map *isl_map_align_range_dims(isl_map *map, int max_dim)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(map != NULL);
    int mdim = isl_map_dim(map, isl_dim_out);
    assert(max_dim >= mdim);

    DEBUG(10, tiramisu::str_dump("Input map:", isl_map_to_str(map)));

    const char *original_range_name = isl_map_get_tuple_name(map, isl_dim_out);

    map = isl_map_add_dims(map, isl_dim_out, max_dim - mdim);

    for (int i = mdim; i < max_dim; i++)
    {
        isl_space *sp = isl_map_get_space(map);
        isl_local_space *lsp = isl_local_space_from_space(sp);
        isl_constraint *cst = isl_constraint_alloc_equality(lsp);
        cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, i, 1);
        map = isl_map_add_constraint(map, cst);
    }

    map = isl_map_set_tuple_name(map, isl_dim_out, original_range_name);

    DEBUG(10, tiramisu::str_dump("After alignment, map = ",
                                 isl_map_to_str(map)));

    DEBUG_INDENT(-4);
    return map;
}

isl_union_map *tiramisu::function::get_aligned_identity_schedules() const
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_union_map *result;
    isl_space *space;

    if (this->body.empty() == false)
    {
        space = isl_map_get_space(this->body[0]->gen_identity_schedule_for_time_space_domain());
    }
    else
    {
        return NULL;
    }
    assert(space != NULL);
    result = isl_union_map_empty(space);

    int max_dim = this->get_max_identity_schedules_range_dim();

    for (const auto &comp : this->get_computations())
    {
        if (comp->should_schedule_this_computation())
        {
            isl_map *sched = comp->gen_identity_schedule_for_time_space_domain();
            DEBUG(3, tiramisu::str_dump("Identity schedule for time space domain: ", isl_map_to_str(sched)));
            assert((sched != NULL) && "Identity schedule could not be computed");
            sched = isl_map_align_range_dims(sched, max_dim);
            result = isl_union_map_union(result, isl_union_map_from_map(sched));
        }
    }

    DEBUG_INDENT(-4);
    DEBUG(3, tiramisu::str_dump("End of function"));

    return result;
}

void tiramisu::function::align_schedules()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int max_dim = this->get_max_schedules_range_dim();

    for (auto &comp : this->get_computations())
    {
        isl_map *dup_sched = comp->get_schedule();
        assert((dup_sched != NULL) && "Schedules should be set before calling align_schedules");
        dup_sched = isl_map_align_range_dims(dup_sched, max_dim);
        comp->set_schedule(dup_sched);
        comp->name_unnamed_time_space_dimensions();
    }

    DEBUG_INDENT(-4);
    DEBUG(3, tiramisu::str_dump("End of function"));
}

void tiramisu::function::add_invariant(tiramisu::constant invar)
{
    invariants.push_back(invar);
}

void tiramisu::function::add_computation(computation *cpt)
{
    assert(cpt != NULL);

    this->body.push_back(cpt);
    if (cpt->should_schedule_this_computation())
        this->starting_computations.insert(cpt);
}

void tiramisu::function::dump(bool exhaustive) const
{
    if (ENABLE_DEBUG)
    {
        std::cout << "\n\nFunction \"" << this->name << "\"" << std::endl << std::endl;

        if (this->function_arguments.size() > 0)
        {
            std::cout << "Function arguments (tiramisu buffers):" << std::endl;
            for (const auto &buf : this->function_arguments)
            {
                buf->dump(exhaustive);
            }
            std::cout << std::endl;
        }

        if (this->invariants.size() > 0)
        {
            std::cout << "Function invariants:" << std::endl;
            for (const auto &inv : this->invariants)
            {
                inv.dump(exhaustive);
            }
            std::cout << std::endl;
        }

        if (this->get_program_context() != NULL)
        {
            std::cout << "Function context set: "
                      << isl_set_to_str(this->get_program_context())
                      << std::endl;
        }

        std::cout << "Body " << std::endl;
        for (const auto &cpt : this->body)
        {
            cpt->dump();
        }
        std::cout << std::endl;

        if (this->halide_stmt.defined())
        {
            std::cout << "Halide stmt " << this->halide_stmt << std::endl;
        }

        std::cout << "Buffers" << std::endl;
        for (const auto &buf : this->buffers_list)
        {
            std::cout << "Buffer name: " << buf.second->get_name() << std::endl;
	    buf.second->dump(false);
        }

        std::cout << std::endl << std::endl;
    }
}

void tiramisu::function::dump_iteration_domain() const
{
    if (ENABLE_DEBUG)
    {
        tiramisu::str_dump("\nIteration domain:\n");
        for (const auto &cpt : this->body)
        {
            cpt->dump_iteration_domain();
        }
        tiramisu::str_dump("\n");
    }
}

void tiramisu::function::dump_schedule() const
{
    if (ENABLE_DEBUG)
    {
        tiramisu::str_dump("\nDumping schedules of the function " + this->get_name() + " :\n");

        for (const auto &cpt : this->body)
        {
            cpt->dump_schedule();
        }

        std::cout << "Parallel dimensions: ";
        for (const auto &par_dim : parallel_dimensions)
        {
            std::cout << par_dim.first << "(" << par_dim.second << ") ";
        }

        std::cout << std::endl;

        std::cout << "Vector dimensions: ";
        for (const auto &vec_dim : vector_dimensions)
        {
            std::cout << std::get<0>(vec_dim) << "(" << std::get<1>(vec_dim) << ") ";
        }

        std::cout << std::endl << std::endl << std::endl;
    }
}

Halide::Argument::Kind halide_argtype_from_tiramisu_argtype(tiramisu::argument_t type)
{
    Halide::Argument::Kind res;

    if (type == tiramisu::a_temporary)
    {
        tiramisu::error("Buffer type \"temporary\" can't be translated to Halide.\n", true);
    }

    if (type == tiramisu::a_input)
    {
        res = Halide::Argument::InputBuffer;
    }
    else
    {
        assert(type == tiramisu::a_output);
        res = Halide::Argument::OutputBuffer;
    }

    return res;
}

void tiramisu::function::set_arguments(const std::vector<tiramisu::buffer *> &buffer_vec)
{
    this->function_arguments = buffer_vec;
}

void tiramisu::function::add_vector_dimension(std::string stmt_name, int vec_dim, int vector_length)
{
    assert(vec_dim >= 0);
    assert(!stmt_name.empty());

    this->vector_dimensions.push_back(std::make_tuple(stmt_name, vec_dim, vector_length));
}

void tiramisu::function::add_distributed_dimension(std::string stmt_name, int dim)
{
    assert(dim >= 0);
    assert(!stmt_name.empty());

    this->distributed_dimensions.push_back({stmt_name, dim});
}

void tiramisu::function::add_parallel_dimension(std::string stmt_name, int vec_dim)
{
    assert(vec_dim >= 0);
    assert(!stmt_name.empty());

    this->parallel_dimensions.push_back({stmt_name, vec_dim});
}

void tiramisu::function::add_unroll_dimension(std::string stmt_name, int level)
{
    assert(level >= 0);
    assert(!stmt_name.empty());

    this->unroll_dimensions.push_back({stmt_name, level});
}

void tiramisu::function::add_gpu_block_dimensions(std::string stmt_name, int dim0,
        int dim1, int dim2)
{
    assert(!stmt_name.empty());
    assert(dim0 >= 0);
    // dim1 and dim2 can be -1 if not set.

    this->gpu_block_dimensions.push_back(
        std::pair<std::string, std::tuple<int, int, int>>(
            stmt_name,
            std::tuple<int, int, int>(dim0, dim1, dim2)));
}

void tiramisu::function::add_gpu_thread_dimensions(std::string stmt_name, int dim0,
        int dim1, int dim2)
{
    assert(!stmt_name.empty());
    assert(dim0 >= 0);
    // dim1 and dim2 can be -1 if not set.

    this->gpu_thread_dimensions.push_back(
        std::pair<std::string, std::tuple<int, int, int>>(
            stmt_name,
            std::tuple<int, int, int>(dim0, dim1, dim2)));
}

isl_union_set *tiramisu::function::get_trimmed_time_processor_domain() const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    isl_union_set *result = NULL;
    isl_space *space = NULL;
    if (!this->body.empty())
    {
        space = isl_set_get_space(this->body[0]->get_trimmed_time_processor_domain());
    }
    else
    {
        DEBUG_INDENT(-4);
        return NULL;
    }
    assert(space != NULL);

    result = isl_union_set_empty(space);

    for (const auto &cpt : this->body)
    {
        if (cpt->should_schedule_this_computation())
        {
            isl_set *cpt_iter_space = isl_set_copy(cpt->get_trimmed_time_processor_domain());
            result = isl_union_set_union(isl_union_set_from_set(cpt_iter_space), result);
        }
    }

    DEBUG_INDENT(-4);

    return result;
}

isl_union_set *tiramisu::function::get_time_processor_domain() const
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_union_set *result = NULL;
    isl_space *space = NULL;

    if (!this->body.empty())
    {
        space = isl_set_get_space(this->body[0]->get_time_processor_domain());
    }
    else
    {
        return NULL;
    }

    assert(space != NULL);
    result = isl_union_set_empty(space);

    for (const auto &cpt : this->body)
    {
        if (cpt->should_schedule_this_computation())
        {
            isl_set *cpt_iter_space = isl_set_copy(cpt->get_time_processor_domain());
            result = isl_union_set_union(isl_union_set_from_set(cpt_iter_space), result);
        }
    }

    DEBUG_INDENT(-4);

    return result;
}

isl_union_set *tiramisu::function::get_iteration_domain() const
{
    isl_union_set *result = NULL;
    isl_space *space = NULL;

    if (!this->body.empty())
    {
        space = isl_set_get_space(this->body[0]->get_iteration_domain());
    }
    else
    {
        return NULL;
    }

    assert(space != NULL);
    result = isl_union_set_empty(space);

    for (const auto &cpt : this->body)
    {
        if (cpt->should_schedule_this_computation())
        {
            isl_set *cpt_iter_space = isl_set_copy(cpt->get_iteration_domain());
            result = isl_union_set_union(isl_union_set_from_set(cpt_iter_space), result);
        }
    }

    return result;
}

isl_union_map *tiramisu::function::get_schedule() const
{
    isl_union_map *result = NULL;
    isl_space *space = NULL;

    if (!this->body.empty())
    {
        space = isl_map_get_space(this->body[0]->get_schedule());
    }
    else
    {
        return NULL;
    }

    assert(space != NULL);
    result = isl_union_map_empty(isl_space_copy(space));

    for (const auto &cpt : this->body)
    {
        isl_map *m = isl_map_copy(cpt->get_schedule());
        result = isl_union_map_union(isl_union_map_from_map(m), result);
    }

    result = isl_union_map_intersect_domain(result, this->get_iteration_domain());

    return result;
}

isl_union_map *tiramisu::function::get_trimmed_schedule() const
{
    isl_union_map *result = NULL;
    isl_space *space = NULL;

    if (!this->body.empty())
    {
        space = isl_map_get_space(this->body[0]->get_trimmed_union_of_schedules());
    }
    else
    {
        return NULL;
    }

    assert(space != NULL);
    result = isl_union_map_empty(isl_space_copy(space));

    for (const auto &cpt : this->body)
    {
        isl_map *m = isl_map_copy(cpt->get_trimmed_union_of_schedules());
        result = isl_union_map_union(isl_union_map_from_map(m), result);
    }

    result = isl_union_map_intersect_domain(result, this->get_iteration_domain());

    return result;
}

// Function for the buffer class

std::string str_tiramisu_type_op(tiramisu::op_t type)
{
    switch (type)
    {
    case tiramisu::o_logical_and:
        return "and";
    case tiramisu::o_logical_or:
        return "or";
    case tiramisu::o_max:
        return "max";
    case tiramisu::o_min:
        return "min";
    case tiramisu::o_minus:
        return "minus";
    case tiramisu::o_add:
        return "add";
    case tiramisu::o_sub:
        return "sub";
    case tiramisu::o_mul:
        return "mul";
    case tiramisu::o_div:
        return "div";
    case tiramisu::o_mod:
        return "mod";
    case tiramisu::o_select:
        return "select";
    case tiramisu::o_lerp:
        return "lerp";
    case tiramisu::o_cond:
        return "ternary_cond";
    case tiramisu::o_logical_not:
        return "not";
    case tiramisu::o_eq:
        return "eq";
    case tiramisu::o_ne:
        return "ne";
    case tiramisu::o_le:
        return "le";
    case tiramisu::o_lt:
        return "lt";
    case tiramisu::o_ge:
        return "ge";
    case tiramisu::o_call:
        return "call";
    case tiramisu::o_access:
        return "access";
    case tiramisu::o_address:
        return "address";
    case tiramisu::o_right_shift:
        return "right-shift";
    case tiramisu::o_left_shift:
        return "left-shift";
    case tiramisu::o_floor:
        return "floor";
    case tiramisu::o_allocate:
        return "allocate";
    case tiramisu::o_free:
        return "free";
    case tiramisu::o_cast:
        return "cast";
    case tiramisu::o_sin:
        return "sin";
    case tiramisu::o_cos:
        return "cos";
    case tiramisu::o_tan:
        return "tan";
    case tiramisu::o_asin:
        return "asin";
    case tiramisu::o_acos:
        return "acos";
    case tiramisu::o_atan:
        return "atan";
    case tiramisu::o_abs:
        return "abs";
    case tiramisu::o_sqrt:
        return "sqrt";
    case tiramisu::o_expo:
        return "exp";
    case tiramisu::o_log:
        return "log";
    case tiramisu::o_ceil:
        return "ceil";
    case tiramisu::o_round:
        return "round";
    case tiramisu::o_trunc:
        return "trunc";
    default:
        tiramisu::error("Tiramisu op not supported.", true);
        return "";
    }
}

const bool tiramisu::buffer::is_allocated() const
{
    return this->allocated;
}

void tiramisu::buffer::mark_as_allocated()
{
    this->allocated = true;
}

std::string str_from_tiramisu_type_expr(tiramisu::expr_t type)
{
    switch (type)
    {
    case tiramisu::e_val:
        return "val";
    case tiramisu::e_op:
        return "op";
    case tiramisu::e_var:
        return "var";
    case tiramisu::e_sync:
        return "sync";
    default:
        tiramisu::error("Tiramisu type not supported.", true);
        return "";
    }
}

std::string str_from_tiramisu_type_argument(tiramisu::argument_t type)
{
    switch (type)
    {
    case tiramisu::a_input:
        return "input";
    case tiramisu::a_output:
        return "output";
    case tiramisu::a_temporary:
        return "temporary";
    default:
        tiramisu::error("Tiramisu type not supported.", true);
        return "";
    }
}

std::string str_from_tiramisu_type_primitive(tiramisu::primitive_t type)
{
    switch (type)
    {
    case tiramisu::p_uint8:
        return "uint8";
    case tiramisu::p_int8:
        return "int8";
    case tiramisu::p_uint16:
        return "uint16";
    case tiramisu::p_int16:
        return "int16";
    case tiramisu::p_uint32:
        return "uin32";
    case tiramisu::p_int32:
        return "int32";
    case tiramisu::p_uint64:
        return "uint64";
    case tiramisu::p_int64:
        return "int64";
    case tiramisu::p_float32:
        return "float32";
    case tiramisu::p_float64:
        return "float64";
    case tiramisu::p_boolean:
        return "bool";
    default:
        tiramisu::error("Tiramisu type not supported.", true);
        return "";
    }
}

std::string str_from_is_null(void *ptr)
{
    return (ptr != NULL) ? "Not NULL" : "NULL";
}

tiramisu::buffer::buffer(std::string name, std::vector<tiramisu::expr> dim_sizes,
                         tiramisu::primitive_t type,
                         tiramisu::argument_t argt, tiramisu::function *fct):
    allocated(false), argtype(argt), auto_allocate(true), dim_sizes(dim_sizes), fct(fct),
    name(name), type(type), location(cuda_ast::memory_location::host)
{
    assert(!name.empty() && "Empty buffer name");
    assert(fct != NULL && "Input function is NULL");

    // Check that the buffer does not already exist.
    assert((fct->get_buffers().count(name) == 0) && ("Buffer already exists"));

    fct->add_buffer(std::pair<std::string, tiramisu::buffer *>(name, this));
};

/**
  * Return the type of the argument (if the buffer is an argument).
  * Three possible types:
  *  - a_input: for inputs of the function,
  *  - a_output: for outputs of the function,
  *  - a_temporary: for buffers used as temporary buffers within
  *  the function (any temporary buffer is allocated automatically by
  *  the Tiramisu runtime at the entry of the function and is
  *  deallocated at the exit of the function).
  */
tiramisu::argument_t buffer::get_argument_type() const
{
    return argtype;
}

/**
  * Return the name of the buffer.
  */
const std::string &buffer::get_name() const
{
    return name;
}

/**
  * Get the number of dimensions of the buffer.
  */
int buffer::get_n_dims() const
{
    return this->get_dim_sizes().size();
}

/**
  * Return the type of the elements of the buffer.
  */
tiramisu::primitive_t buffer::get_elements_type() const
{
    return type;
}

/**
  * Return the sizes of the dimensions of the buffer.
  * Assuming the following buffer: buf[N0][N1][N2].  The first
  * vector element represents the size of rightmost dimension
  * of the buffer (i.e. N2), the second vector element is N1,
  * and the last vector element is N0.
  */
const std::vector<tiramisu::expr> &buffer::get_dim_sizes() const
{
    return dim_sizes;
}

void tiramisu::buffer::dump(bool exhaustive) const
{
    if (ENABLE_DEBUG)
    {
        std::cout << "Buffer \"" << this->name
                  << "\", Number of dimensions: " << this->get_n_dims()
                  << std::endl;

        std::cout << "Dimension sizes: ";
        for (const auto &size : dim_sizes)
		size.dump(false);
        std::cout << std::endl;

        std::cout << "Elements type: "
                  << str_from_tiramisu_type_primitive(this->type) << std::endl;

        std::cout << "Function field: "
                  << str_from_is_null(this->fct) << std::endl;

        std::cout << "Argument type: "
                  << str_from_tiramisu_type_argument(this->argtype) << std::endl;

        std::cout << std::endl << std::endl;
    }
}

Halide::Type halide_type_from_tiramisu_type(tiramisu::primitive_t type)
{
    Halide::Type t;

    switch (type)
    {
    case tiramisu::p_uint8:
        t = Halide::UInt(8);
        break;
    case tiramisu::p_int8:
        t = Halide::Int(8);
        break;
    case tiramisu::p_uint16:
        t = Halide::UInt(16);
        break;
    case tiramisu::p_int16:
        t = Halide::Int(16);
        break;
    case tiramisu::p_uint32:
        t = Halide::UInt(32);
        break;
    case tiramisu::p_int32:
        t = Halide::Int(32);
        break;
    case tiramisu::p_uint64:
        t = Halide::UInt(64);
        break;
    case tiramisu::p_int64:
        t = Halide::Int(64);
        break;
    case tiramisu::p_float32:
        t = Halide::Float(32);
        break;
    case tiramisu::p_float64:
        t = Halide::Float(64);
        break;
    case tiramisu::p_boolean:
        t = Halide::Bool();
        break;
    default:
        tiramisu::error("Tiramisu type cannot be translated to Halide type.", true);
    }
    return t;
}

//----------------

std::map<std::string, isl_ast_expr *> tiramisu::computation::get_iterators_map()
{
    return this->iterators_map;
}

void tiramisu::computation::set_iterators_map(std::map<std::string, isl_ast_expr *> map)
{
    this->iterators_map = map;
}

tiramisu::expr tiramisu::computation::get_predicate()
{
    return this->predicate;
}

void tiramisu::computation::add_predicate(tiramisu::expr predicate)
{
    this->predicate = predicate;
}

/**
  * Initialize a computation
  *  This is a private function that should not be called explicitly
  * by users.
  */
void tiramisu::computation::init_computation(std::string iteration_space_str,
        tiramisu::function *fction,
        const tiramisu::expr &e,
        bool schedule_this_computation,
        tiramisu::primitive_t t)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Constructing the computation: " + iteration_space_str));

    assert(fction != NULL);
    assert(iteration_space_str.length() > 0 && ("Empty iteration space"));

    // Initialize all the fields to NULL (useful for later asserts)
    access = NULL;
    stmt = Halide::Internal::Stmt();
    time_processor_domain = NULL;
    duplicate_number = 0;
    automatically_allocated_buffer = NULL;
    predicate = tiramisu::expr();
    // In the constructor of computations, we assume that every created
    // computation is the first computation, then, if this computation
    // was created by add_definitions(), we change is_first_definition
    // to false, otherwise we keep it.
    // We do the same for first_definition.
    is_first = true;
    first_definition = NULL;
    this->definitions_number = 1;
    this->definition_ID = 0;
    this->_is_library_call = false;
    this->_is_nonblock_or_async = false;
    this->_drop_rank_iter = false;

    this->lhs_access_type = tiramisu::o_access;
    this->lhs_argument_idx = -1;
    this->rhs_argument_idx = -1;
    this->wait_argument_idx = -1;
    this->_is_library_call = false;
    this->wait_access_map = nullptr;
    this->wait_index_expr = nullptr;

    this->schedule_this_computation = schedule_this_computation;
    this->data_type = t;

    this->ctx = fction->get_isl_ctx();

    iteration_domain = isl_set_read_from_str(ctx, iteration_space_str.c_str());
    this->name_unnamed_iteration_domain_dimensions();
    name = std::string(isl_space_get_tuple_name(isl_set_get_space(iteration_domain),
                       isl_dim_type::isl_dim_set));

    number_of_dims = isl_set_dim(iteration_domain, isl_dim_type::isl_dim_set);
    for (unsigned i = 0; i < number_of_dims; i++) {
        if (isl_set_has_dim_name(iteration_domain, isl_dim_type::isl_dim_set, i)) {
            std::string dim_name(isl_set_get_dim_name(iteration_domain, isl_dim_type::isl_dim_set, i));
            this->access_variables.push_back(make_pair(i, dim_name));
        }
    }

    fct = fction;
    fct->add_computation(this);
    this->set_identity_schedule_based_on_iteration_domain();
    this->set_expression(e);
    this->set_inline(false);

    // Set the names of output dimensions to be equal to the names of iteration domain schedules.
    std::vector<std::string> nms = this->get_iteration_domain_dimension_names();
    // Rename the dimensions of the schedule domain so that when we set the names of
    // the schedule range dimension to be equal to the names of the domain, we do not
    // get a conflict.
    for (int i = 0; i< this->get_iteration_domain_dimensions_number(); i++)
	this->set_schedule_domain_dim_names({i}, {generate_new_variable_name()});
    for (int i = 0; i< nms.size(); i++)
    	this->set_loop_level_names({i}, {nms[i]});

    // If there are computations that have already been defined and that
    // have the same name, check that they have constraints over their iteration
    // domains.
    std::vector<tiramisu::computation *> same_name_computations =
        this->get_function()->get_computation_by_name(name);
    if (same_name_computations.size() > 1)
    {
        if (isl_set_plain_is_universe(this->get_iteration_domain()))
            tiramisu::error("Computations defined multiple times should"
                            " have bounds on their iteration domain", true);

        for (auto c : same_name_computations)
        {
            if (isl_set_plain_is_universe(c->get_iteration_domain()))
                tiramisu::error("Computations defined multiple times should"
                                " have bounds on their iteration domain", true);
        }
    }

    this->updates.push_back(this);

    DEBUG_INDENT(-4);
}

/**
 * Dummy constructor for derived classes.
 */
tiramisu::computation::computation()
{
    this->access = NULL;
    this->schedule = NULL;
    this->stmt = Halide::Internal::Stmt();
    this->time_processor_domain = NULL;
    this->duplicate_number = 0;

    this->schedule_this_computation = false;
    this->data_type = p_none;
    this->expression = tiramisu::expr();

    this->ctx = NULL;

    this->lhs_access_type = tiramisu::o_access;
    this->lhs_argument_idx = -1;
    this->rhs_argument_idx = -1;
    this->wait_argument_idx = -1;
    this->_is_library_call = false;
    this->wait_access_map = nullptr;
    this->wait_index_expr = nullptr;

    this->iteration_domain = NULL;
    this->name = "";
    this->fct = NULL;
    this->is_let = false;
}

/**
  * Constructor for computations.
  *
  * \p iteration_domain_str is a string that represents the iteration
  * domain of the computation.  The iteration domain should be written
  * in the ISL format (http://isl.gforge.inria.fr/user.html#Sets-and-Relations).
  *
  * The iteration domain of a statement is a set that contains
  * all of the execution instances of the statement (a statement in a
  * loop has an execution instance for each loop iteration in which
  * it executes). Each execution instance of a statement in a loop
  * nest is uniquely represented by an identifier and a tuple of
  * integers  (typically,  the  values  of  the  outer  loop  iterators).
  *
  * For example, the iteration space of the statement S0 in the following
  * loop nest
  * for (i=0; i<2; i++)
  *   for (j=0; j<3; j++)
  *      S0;
  *
  * is {S0(0,0), S0(0,1), S0(0,2), S0(1,0), S0(1,1), S0(1,2)}
  *
  * S0(0,0) is the execution instance of S0 in the iteration (0,0).
  *
  * The previous set of integer tuples can be compactly described
  * by affine constraints as follows
  *
  * {S0(i,j): 0<=i<2 and 0<=j<3}
  *
  * In general, the loop nest
  *
  * for (i=0; i<N; i++)
  *   for (j=0; j<M; j++)
  *      S0;
  *
  * has the following iteration domain
  *
  * {S0(i,j): 0<=i<N and 0<=j<M}
  *
  * This should be read as: the set of points (i,j) such that
  * 0<=i<N and 0<=j<M.
  *
  * \p e is the expression computed by the computation.
  *
  * \p schedule_this_computation should be set to true if the computation
  * is supposed to be schedule and code is supposed to be generated from
  * the computation.  Set it to false if you just want to use the
  * computation to represent a buffer (that is passed as an argument
  * to the function) and you do not intend to generate code for the
  * computation.
  *
  * \p t is the type of the computation, i.e. the type of the expression
  * computed by the computation. Example of types include (p_uint8,
  * p_uint16, p_uint32, ...).
  *
  * \p fct is a pointer to the Tiramisu function where this computation
  * should be added.
  *
  * TODO: copy ISL format for sets.
  */
tiramisu::computation::computation(std::string iteration_domain_str, tiramisu::expr e,
                                   bool schedule_this_computation, tiramisu::primitive_t t,
                                   tiramisu::function *fct)
{

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    init_computation(iteration_domain_str, fct, e, schedule_this_computation, t);
    is_let = false;

    DEBUG_INDENT(-4);
}

/**
  * Return true if the this computation is supposed to be scheduled
  * by Tiramisu.
  */
bool tiramisu::computation::should_schedule_this_computation() const
{
    return schedule_this_computation;
}

/**
  * Return the access function of the computation.
  */
isl_map *tiramisu::computation::get_access_relation() const
{
    return access;
}

/**
  * Return the access function of the computation after transforming
  * it to the time-processor domain.
  * The domain of the access function is transformed to the
  * time-processor domain using the schedule, and then the transformed
  * access function is returned.
  */
isl_map *tiramisu::computation::get_access_relation_adapted_to_time_processor_domain() const
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(10, tiramisu::str_dump("Getting the access of the computation " + this->get_name() +
                                 " adapted to time-space."));
    assert((this->has_accesses() == true) && ("This computation must have accesses."));

    isl_map *access = isl_map_copy(this->get_access_relation());

    if (!this->is_let_stmt())
    {
        DEBUG(10, tiramisu::str_dump("Original access:", isl_map_to_str(access)));

        if (global::is_auto_data_mapping_set())
        {
            if (access != NULL)
            {
                assert(this->get_trimmed_union_of_schedules() != NULL);

                DEBUG(10, tiramisu::str_dump("Original schedule:", isl_map_to_str(this->get_schedule())));
                DEBUG(10, tiramisu::str_dump("Trimmed schedule to apply:",
                                             isl_map_to_str(this->get_trimmed_union_of_schedules())));
                access = isl_map_apply_domain(
                             isl_map_copy(access),
                             isl_map_copy(this->get_trimmed_union_of_schedules()));
                DEBUG(10, tiramisu::str_dump("Transformed access:", isl_map_to_str(access)));
            }
            else
            {
                DEBUG(10, tiramisu::str_dump("Not access relation to transform."));
            }
        }
        else
        {
            DEBUG(10, tiramisu::str_dump("Access not transformed"));
        }
    }
    else
    {
        DEBUG(10, tiramisu::str_dump("This is a let statement."));
    }

    DEBUG_INDENT(-4);

    return access;
}

/**
 * Return the Tiramisu expression associated with the computation.
 */
const tiramisu::expr &tiramisu::computation::get_expr() const
{
    return expression;
}

tiramisu::computation::operator expr()
{
    // assert(this->is_let_stmt() && "Can only use let statements as expressions.");
    return var(this->get_data_type(), this->get_name());
}

/**
  * Return the function where the computation is declared.
  */
tiramisu::function *tiramisu::computation::get_function() const
{
    return fct;
}

/**
  * Return vector of isl_ast_expr representing the indices of the array where
  * the computation will be stored.
  */
std::vector<isl_ast_expr *> &tiramisu::computation::get_index_expr()
{
    return index_expr;
}

/**
  * Return the iteration domain of the computation.
  * In this representation, the order of execution of computations
  * is not specified, the computations are also not mapped to memory.
  */
isl_set *tiramisu::computation::get_iteration_domain() const
{
    // Every computation should have an iteration space.
    assert(iteration_domain != NULL);

    return iteration_domain;
}

computation * computation::get_predecessor() {
    auto &reverse_graph = this->get_function()->sched_graph_reversed[this];

    if (reverse_graph.empty())
        return nullptr;

    return reverse_graph.begin()->first;
}

/**
  * Return the time-processor domain of the computation.
  * In this representation, the logical time of execution and the
  * processor where the computation will be executed are both
  * specified.
  */
isl_set *tiramisu::computation::get_time_processor_domain() const
{
    return time_processor_domain;
}

/**
  * Return the trimmed schedule of the computation.
  * The trimmed schedule is the schedule without the
  * duplication dimension.
  */
isl_map *tiramisu::computation::get_trimmed_union_of_schedules() const
{
    isl_map *trimmed_sched = isl_map_copy(this->get_schedule());
    const char *name = isl_map_get_tuple_name(this->get_schedule(), isl_dim_out);
    trimmed_sched = isl_map_project_out(trimmed_sched, isl_dim_out, 0, 1);
    trimmed_sched = isl_map_set_tuple_name(trimmed_sched, isl_dim_out, name);

    return trimmed_sched;
}

/**
 * Return if this computation represents a let statement.
 */
bool tiramisu::computation::is_let_stmt() const
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::string s1 = "This computation is ";
    std::string s2 = (is_let?" a ":" not a ");
    std::string s3 = "let statement.";

    DEBUG(10, tiramisu::str_dump(s1 + s2 + s3));

    DEBUG_INDENT(-4);

    return is_let;
}

bool tiramisu::computation::is_library_call() const
{
    return this->_is_library_call;
}

bool tiramisu::computation::should_drop_rank_iter() const
{
    return this->_drop_rank_iter;
}

/**
  * Return the name of the computation.
  */
const std::string &tiramisu::computation::get_name() const
{
    return name;
}

/**
  * Return a unique name of computation; made of the following pattern:
  * [computation name]@[computation address in memory]
  */
const std::string tiramisu::computation::get_unique_name() const
{
    std::stringstream namestream;
    namestream << get_name();
    namestream << "@";
    namestream << (void *)this;
    return namestream.str();
}

/**
  * Return the context of the computations.
  */
isl_ctx *tiramisu::computation::get_ctx() const
{
    return ctx;
}

/**
 * Get the number of dimensions of the iteration
 * domain of the computation.
 */
int tiramisu::computation::get_iteration_domain_dimensions_number()
{
    assert(iteration_domain != NULL);

    return isl_set_n_dim(this->iteration_domain);
}

int tiramisu::computation::get_time_space_dimensions_number()
{
    assert(this->get_schedule() != NULL);

    return isl_map_dim(this->get_schedule(), isl_dim_out);
}

int computation::get_loop_levels_number()
{
    assert(this->get_schedule() != NULL);
    int loop_levels_number = ((isl_map_dim(this->get_schedule(), isl_dim_out)) - 2)/2;

    return loop_levels_number;
}

/**
 * Get the data type of the computation.
 */
tiramisu::primitive_t tiramisu::computation::get_data_type() const
{
    return data_type;
}

/**
  * Return the Halide statement that assigns the computation to a buffer location.
  */
Halide::Internal::Stmt tiramisu::computation::get_generated_halide_stmt() const
{
    return stmt;
}

/**
 * Compare two computations.
 *
 * Two computations are considered to be equal if they have the
 * same name.
 */
bool tiramisu::computation::operator==(tiramisu::computation comp1)
{
    return (this->get_name() == comp1.get_name());
}

/**
  * Generate the time-processor domain of the computation.
  *
  * In this representation, the logical time of execution and the
  * processor where the computation will be executed are both
  * specified.  The memory location where computations will be
  * stored in memory is not specified at the level.
  */
void tiramisu::computation::gen_time_space_domain()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(this->get_iteration_domain() != NULL);
    assert(this->get_schedule() != NULL);

    DEBUG(3, tiramisu::str_dump("Iteration domain:", isl_set_to_str(this->get_iteration_domain())));

    isl_set *iter = isl_set_copy(this->get_iteration_domain());
    iter = this->intersect_set_with_context(iter);

    DEBUG(3, tiramisu::str_dump("Iteration domain Intersect context:", isl_set_to_str(iter)));

    time_processor_domain = isl_set_apply(
                                iter,
                                isl_map_copy(this->get_schedule()));

    DEBUG(3, tiramisu::str_dump("Schedule:", isl_map_to_str(this->get_schedule())));
    DEBUG(3, tiramisu::str_dump("Generated time-space domain:", isl_set_to_str(time_processor_domain)));

    DEBUG_INDENT(-4);
}

void tiramisu::computation::drop_rank_iter()
{
    this->_drop_rank_iter = true;
}

void tiramisu::computation::set_access(isl_map *access)
{
    assert(access != NULL);

    this->set_access(isl_map_to_str(access));
}

/**
 * Set the access function of the computation.
 *
 * The access function is a relation from computations to buffer locations.
 * \p access_str is a string that represents the relation (in ISL format,
 * http://isl.gforge.inria.fr/user.html#Sets-and-Relations).
 */
void tiramisu::computation::set_access(std::string access_str)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Setting access " + access_str + " for computation " + this->get_name()));

    this->access = isl_map_read_from_str(this->ctx, access_str.c_str());

    /**
     * Set the access relations of all the computations that have the same name
     * (duplicates and updates).
     */
    std::vector<tiramisu::computation *> same_name_computations =
        this->get_function()->get_computation_by_name(this->get_name());

    if (same_name_computations.size() > 1)
        for (auto c : same_name_computations)
        {
            c->access = isl_map_read_from_str(this->ctx, access_str.c_str());
        }

    /**
     * Check that if there are other computations that have the same name
     * as this computation, then the access of all of these computations
     * should be the same.
     */
    std::vector<tiramisu::computation *> computations =
        this->get_function()->get_computation_by_name(this->get_name());
    for (auto c : computations)
        if (isl_map_is_equal(this->get_access_relation(), c->get_access_relation()) == isl_bool_false)
        {
            tiramisu::error("Computations that have the same name should also have the same access relation.",
                            true);
        }

    assert(this->access != nullptr && "Set access failed");

    DEBUG_INDENT(-4);
}

/**
 * Generate an identity schedule for the computation.
 *
 * This identity schedule is an identity relation created from the iteration
 * domain.
 */
isl_map *tiramisu::computation::gen_identity_schedule_for_iteration_domain()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_space *sp = isl_set_get_space(this->get_iteration_domain());
    isl_map *sched = isl_map_identity(isl_space_map_from_set(sp));
    sched = isl_map_intersect_domain(sched, isl_set_copy(this->get_iteration_domain()));
    sched = isl_map_coalesce(sched);

    // Add Beta dimensions.
    for (int i = 0; i < isl_space_dim(sp, isl_dim_out) + 1; i++)
    {
        sched = isl_map_add_dim_and_eq_constraint(sched, 2 * i, 0);
    }

    // Add the duplication dimension.
    sched = isl_map_add_dim_and_eq_constraint(sched, 0, 0);

    DEBUG_INDENT(-4);

    return sched;
}

isl_set *tiramisu::computation::get_trimmed_time_processor_domain()
{
    isl_set *tp_domain = isl_set_copy(this->get_time_processor_domain());
    const char *name = isl_set_get_tuple_name(isl_set_copy(tp_domain));
    isl_set *tp_domain_without_duplicate_dim =
        isl_set_project_out(isl_set_copy(tp_domain), isl_dim_set, 0, 1);
    tp_domain_without_duplicate_dim = isl_set_set_tuple_name(tp_domain_without_duplicate_dim, name);
    return tp_domain_without_duplicate_dim ;
}

/**
 * Generate an identity schedule for the computation.
 *
 * This identity schedule is an identity relation created from the
 * time-processor domain.  It removes the "duplicate" dimension (i.e.,
 * the dimension used to identify duplicate computations).
 */
isl_map *tiramisu::computation::gen_identity_schedule_for_time_space_domain()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_set *tp_domain = this->get_trimmed_time_processor_domain();
    isl_space *sp = isl_set_get_space(tp_domain);
    isl_map *sched = isl_map_identity(isl_space_map_from_set(sp));
    sched = isl_map_intersect_domain(
                sched, isl_set_copy(this->get_trimmed_time_processor_domain()));
    sched = isl_map_set_tuple_name(sched, isl_dim_out, "");
    sched = isl_map_coalesce(sched);

    DEBUG_INDENT(-4);

    return sched;
}

/**
 * Set an identity schedule for the computation.
 *
 * This identity schedule is an identity relation created from the iteration
 * domain.
 */
void tiramisu::computation::set_identity_schedule_based_on_iteration_domain()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_map *sched = this->gen_identity_schedule_for_iteration_domain();
    DEBUG(3, tiramisu::str_dump("The following identity schedule is generated (setting schedule 0): "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(sched)));
    this->set_schedule(sched);
    DEBUG(3, tiramisu::str_dump("The identity schedule for the original computation is set."));

    DEBUG_INDENT(-4);
}

std::vector<std::string> computation::get_loop_level_names()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Collecting names of loop levels from the range of the schedule: ", isl_map_to_str(this->get_schedule())));

    std::vector<std::string> names;
    std::string names_to_print_for_debugging = "";

    for (int i = 0; i < this->get_loop_levels_number(); i++)
    {
	std::string dim_name = isl_map_get_dim_name(this->get_schedule(), isl_dim_out, loop_level_into_dynamic_dimension(i));
	names.push_back(dim_name);
	names_to_print_for_debugging += dim_name + " ";
    }

    DEBUG(3, tiramisu::str_dump("Names of loop levels: " + names_to_print_for_debugging));
    DEBUG_INDENT(-4);

    return names;
}

int computation::get_duplicates_number() const
{
    return this->duplicate_number;
}

isl_map *computation::get_schedule() const
{
    return this->schedule;
}

void tiramisu::computation::add_associated_let_stmt(std::string variable_name, tiramisu::expr e)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!variable_name.empty());
    assert(e.is_defined());

    DEBUG(3, tiramisu::str_dump("Adding a let statement associated to the computation " +
                                this->get_name() + "."));
    DEBUG(3, tiramisu::str_dump("The name of the variable of the let statement: " + variable_name +
                                "."));
    DEBUG(3, tiramisu::str_dump("Expression: ")); e.dump(false);

    this->associated_let_stmts.push_back({variable_name, e});

    DEBUG_INDENT(-4);
}

bool tiramisu::computation::is_send() const
{
  return false;
}

bool tiramisu::computation::is_recv() const
{
  return false;
}

bool tiramisu::computation::is_send_recv() const
{
  return false;
}

bool tiramisu::computation::is_wait() const
{
  return false;
}

const std::vector<std::pair<std::string, tiramisu::expr>>
        &tiramisu::computation::get_associated_let_stmts() const
{
    return this->associated_let_stmts;
}

bool tiramisu::computation::has_accesses() const
{
    if ((this->get_expr().get_op_type() == tiramisu::o_access))
        return true;
    else if ((this->get_expr().get_op_type() == tiramisu::o_allocate) ||
            (this->get_expr().get_op_type() == tiramisu::o_free) ||
            (this->get_expr().get_op_type() == tiramisu::o_memcpy) ||
            (this->get_expr().get_expr_type() == tiramisu::e_sync) ||
            (this->is_let_stmt()))
    {
        return false;
    }
    else
    {
        return true;
    }
}

/**
 * Set the expression of the computation.
 */
void tiramisu::computation::set_expression(const tiramisu::expr &e)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("The original expression is: "));
    e.dump(false);
    DEBUG(3, tiramisu::str_dump(""));

    DEBUG(3, tiramisu::str_dump("Traversing the expression to replace non-affine accesses by a constant definition."));
    tiramisu::expr modified_e = traverse_expr_and_replace_non_affine_accesses(this, e);

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("The new expression is: "); modified_e.dump(false););
    DEBUG(3, tiramisu::str_dump(""));

    this->expression = modified_e.copy();

    DEBUG_INDENT(-4);
}

void tiramisu::computation::set_inline(bool is_inline) {
    this->is_inline = is_inline;
}

const bool tiramisu::computation::is_inline_computation() const {
    return this->is_inline;
}

/**
 * Set the name of the computation.
 */
void tiramisu::computation::set_name(const std::string &n)
{
    this->name = n;
}

/**
  * Bind the computation to a buffer.
  * i.e. create a one-to-one data mapping between the computation
  * the buffer.
  */
void tiramisu::computation::bind_to(buffer *buff)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(buff != NULL);

    isl_space *sp = isl_set_get_space(this->get_iteration_domain());
    isl_map *map = isl_map_identity(isl_space_map_from_set(sp));
    map = isl_map_set_tuple_name(map, isl_dim_out, buff->get_name().c_str());
    map = isl_map_coalesce(map);

    DEBUG(3, tiramisu::str_dump("Binding. The following access function is set: ",
                                isl_map_to_str(map)));

    this->set_access(isl_map_to_str(map));

    isl_map_free(map);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::mark_as_let_statement()
{
    this->is_let = true;
}

void tiramisu::computation::mark_as_library_call()
{
    this->_is_library_call = true;
}

/****************************************************************************
 ****************************************************************************
 ***************************** Constant class *******************************
 ****************************************************************************
 ****************************************************************************/

tiramisu::constant::constant(
    std::string param_name, const tiramisu::expr &param_expr,
    tiramisu::primitive_t t,
    tiramisu::function *func): tiramisu::computation()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(!param_name.empty() && "Parameter name empty");
    assert((func != NULL) && "Function undefined");

    DEBUG(3, tiramisu::str_dump("Constructing a scheduled, function-wide constant (this is supposed to replace non-scheduled function wide computations."));

    this->set_name(param_name);
    this->set_expression(param_expr);
    this->mark_as_let_statement();
    this->compute_with_computation = NULL;
    DEBUG(3, tiramisu::str_dump("The constant is function wide, but it is scheduled.  Its name is : "));
    DEBUG(3, tiramisu::str_dump(this->get_name()));
    std::string iter_str = "{" + this->get_name() + "[0]}";
    DEBUG(3, tiramisu::str_dump("Computed iteration space for the constant assignment" + iter_str));
    init_computation(iter_str, func, param_expr, true, t);
    DEBUG_NO_NEWLINE(10, tiramisu::str_dump("The computation representing the assignment:"); this->dump(true));

    DEBUG_INDENT(-4);
}


tiramisu::constant::constant(
    std::string param_name, const tiramisu::expr &param_expr,
    tiramisu::primitive_t t,
    bool function_wide,
    tiramisu::computation *with_computation,
    int at_loop_level,
    tiramisu::function *func): tiramisu::computation()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(!param_name.empty() && "Parameter name empty");
    assert((func != NULL) && "Function undefined");
    assert((((function_wide == true) && (with_computation == NULL)) || ((function_wide == false) && (with_computation != NULL))) &&
           "with_computation, should be set only if function_wide is false");

    DEBUG(3, tiramisu::str_dump("Constructing a constant."));

    if (function_wide)
    {
        this->set_name(param_name);
        this->set_expression(param_expr);
        this->mark_as_let_statement();
        this->data_type = t;
        func->add_invariant(*this);
        this->compute_with_computation = NULL;

        DEBUG(3, tiramisu::str_dump("The constant is function wide, its name is : "));
        DEBUG(3, tiramisu::str_dump(this->get_name()));
    }
    else
    {
        assert((with_computation != NULL) &&
               "A valid computation should be provided.");
        assert((at_loop_level >= computation::root_dimension) &&
               "Invalid root dimension.");

	DEBUG(3, tiramisu::str_dump("Consturcting constant at level: " + std::to_string(at_loop_level)));

	this->compute_with_computation = with_computation;
        isl_set *iter = with_computation->get_iteration_domain();
        int projection_dimension = at_loop_level + 1;
        iter = isl_set_project_out(isl_set_copy(iter),
                                   isl_dim_set,
                                   projection_dimension,
                                   isl_set_dim(iter, isl_dim_set) - projection_dimension);
        iter = isl_set_set_tuple_name(iter, param_name.c_str());
        std::string iteration_domain_str = isl_set_to_str(iter);

        DEBUG(3, tiramisu::str_dump(
                  "Computed iteration space for the constant assignment",
                  isl_set_to_str(iter)));

        init_computation(iteration_domain_str, func, param_expr, true, t);

        this->mark_as_let_statement();

        DEBUG_NO_NEWLINE(10,
                         tiramisu::str_dump("The computation representing the assignment:");
                         this->dump(true));

        // Set the schedule of this computation to be executed
        // before the computation.
	if (with_computation->get_predecessor() != NULL)
	    this->between(*(with_computation->get_predecessor()),
			  this->get_dimension_name_for_loop_level(at_loop_level),
			  *with_computation,
			  this->get_dimension_name_for_loop_level(at_loop_level));
	else
	    this->before(*with_computation, at_loop_level);

        DEBUG(3, tiramisu::str_dump("The constant is not function wide, the iteration domain of the constant is: "));
        DEBUG(3, tiramisu::str_dump(isl_set_to_str(this->get_iteration_domain())));
    }

    DEBUG_INDENT(-4);
}

void tiramisu::constant::dump(bool exhaustive) const
{
    if (ENABLE_DEBUG)
    {
        std::cout << "Invariant \"" << this->get_name() << "\"" << std::endl;
        std::cout << "Expression: ";
        this->get_expr().dump(false);
        std::cout << std::endl;
    }
}

tiramisu::constant::operator expr()
{
    return var(this->get_data_type(), this->get_name());
    // return this->get_expr();
}

void tiramisu::buffer::set_dim_size(int dim, int size)
{
    assert(dim >= 0);
    assert(dim < this->dim_sizes.size());
    assert(this->dim_sizes.size() > 0);
    assert(size > 0);

    this->dim_sizes[dim] = size;
}

void tiramisu::computation::storage_fold(tiramisu::var L0_var, int factor)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    std::vector<int> loop_dimensions =
	this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(loop_dimensions);
    int inDim0 = loop_dimensions[0];

    assert(this->get_access_relation() != NULL);
    assert(inDim0 >= 0);
    assert(inDim0 < isl_space_dim(isl_map_get_space(this->get_access_relation()), isl_dim_out));
    assert(factor > 0);

    isl_map *access_relation = this->get_access_relation();
    std::string buffer_name = isl_map_get_tuple_name(access_relation, isl_dim_out);
    tiramisu::buffer *buff_object = this->get_function()->get_buffers().find(buffer_name)->second;
    buff_object->set_dim_size(inDim0, factor);

    access_relation = isl_map_copy(access_relation);

    DEBUG(3, tiramisu::str_dump("Original access relation: ", isl_map_to_str(access_relation)));
    DEBUG(3, tiramisu::str_dump("Folding dimension " + std::to_string(inDim0)
                                + " by a factor of " + std::to_string(factor)));

    std::string inDim0_str;

    std::string outDim0_str = generate_new_variable_name();

    int n_dims = isl_map_dim(access_relation, isl_dim_out);
    std::vector<isl_id *> dimensions;
    std::vector<std::string> dimensions_str;
    std::string map = "{";

    // -----------------------------------------------------------------
    // Preparing a map to split the duplicate computation.
    // -----------------------------------------------------------------

    map = map + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        std::string dim_str = generate_new_variable_name();
        dimensions_str.push_back(dim_str);
        map = map + dim_str;

        if (i == inDim0)
        {
            inDim0_str = dim_str;
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] -> " + buffer_name + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i != inDim0)
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else
        {
            map = map + outDim0_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim0_str.c_str(), NULL);
            dimensions.push_back(id0);
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] : " + outDim0_str + " = floor(" + inDim0_str + "%" +
          std::to_string(factor) + ")}";

    isl_map *transformation_map = isl_map_read_from_str(this->get_ctx(), map.c_str());

    for (int i = 0; i < dimensions.size(); i++)
        transformation_map = isl_map_set_dim_id(
                                 transformation_map, isl_dim_out, i, isl_id_copy(dimensions[i]));

    transformation_map = isl_map_set_tuple_id(
                             transformation_map, isl_dim_in,
                             isl_map_get_tuple_id(isl_map_copy(access_relation), isl_dim_out));

    isl_id *id_range = isl_id_alloc(this->get_ctx(), buffer_name.c_str(), NULL);
    transformation_map = isl_map_set_tuple_id(transformation_map, isl_dim_out, id_range);

    DEBUG(3, tiramisu::str_dump("Transformation map : ",
                                isl_map_to_str(transformation_map)));

    access_relation = isl_map_apply_range(isl_map_copy(access_relation),
                                          isl_map_copy(transformation_map));

    DEBUG(3, tiramisu::str_dump("Access relation after storage folding: ",
                                isl_map_to_str(access_relation)));

    this->set_access(access_relation);



    DEBUG_INDENT(-4);
}

void tiramisu::computation::set_parent_computation(tiramisu::computation *parent_computation) {
    this->parent_computation = parent_computation;
}

std::set<int> tiramisu::xfer_prop::xfer_prop_ids;

tiramisu::xfer_prop::xfer_prop() { }

tiramisu::xfer_prop::xfer_prop(tiramisu::primitive_t dtype,
                               std::initializer_list<tiramisu::xfer_attr> attrs)
        : dtype(dtype), xfer_prop_id(-1) {
    this->attrs.insert(this->attrs.begin(), attrs);
}

tiramisu::xfer_prop::xfer_prop(tiramisu::primitive_t dtype,
                               std::initializer_list<tiramisu::xfer_attr> attrs,
                               int comm_prop_id) : dtype(dtype), xfer_prop_id(comm_prop_id) {
    this->attrs.insert(this->attrs.begin(), attrs);
    if (comm_prop_id != -1) {
        xfer_prop_ids.insert(comm_prop_id);
    }
    xfer_prop_ids.insert(0); // The kernel one. Just make sure it gets in there
}

tiramisu::primitive_t tiramisu::xfer_prop::get_dtype() const {
    return this->dtype;
}

std::string tiramisu::xfer_prop::attr_to_string(tiramisu::xfer_attr attr) {
    switch (attr) {
        case SYNC: return "SYNC";
        case ASYNC: return "ASYNC";
        case MPI: return "MPI";
        case CUDA: return "CUDA";
        case BLOCK: return "BLOCK";
        case NONBLOCK: return "NONBLOCK";
        default: {
            assert(false && "Unknown xfer_prop attr specified.");
            return "";
        }
    }
}

int tiramisu::xfer_prop::get_xfer_prop_id() const {
    return xfer_prop_id;
}

void tiramisu::xfer_prop::add_attr(tiramisu::xfer_attr attr) {
    attrs.push_back(attr);
}

bool tiramisu::xfer_prop::contains_attr(tiramisu::xfer_attr attr) const {
    return attrs.end() != std::find(attrs.begin(), attrs.end(), attr);
}

bool tiramisu::xfer_prop::contains_attrs(std::vector<tiramisu::xfer_attr> attrs) const {
    for (auto attr : attrs) {
        if (this->attrs.end() == std::find(this->attrs.begin(), this->attrs.end(), attr)) {
            return false;
        }
    }
    return true;
}

tiramisu::communicator::communicator() { }

tiramisu::communicator::communicator(std::string iteration_domain_str, tiramisu::expr e,
                                     bool schedule_this_computation, tiramisu::primitive_t data_type,
                                     tiramisu::xfer_prop prop, tiramisu::function *fct) :
        computation(iteration_domain_str, e, schedule_this_computation, data_type, fct), prop(prop) {}

tiramisu::communicator::communicator(std::string iteration_domain_str, tiramisu::expr e, bool schedule_this_computation,
                                     tiramisu::primitive_t data_type, tiramisu::function *fct) :
        computation(iteration_domain_str, e, schedule_this_computation, data_type, fct) {}

void tiramisu::communicator::collapse_many(std::vector<tiramisu::collapse_group> collapse_each) {
    for (auto c : collapse_each) {
        this->collapse(std::get<0>(c), std::get<1>(c), -1, std::get<2>(c));
    }
}

void tiramisu::communicator::add_dim(tiramisu::expr dim)
{
    this->dims.push_back(dim);
}

tiramisu::expr tiramisu::communicator::get_num_elements() const
{
    tiramisu::expr num = expr(1);
    if (!dims.empty()) {
        num = tiramisu::expr(tiramisu::o_cast, dims[0].get_data_type(), num);
    }
    for (std::vector<tiramisu::expr>::const_iterator iter = dims.cbegin(); iter != dims.cend(); iter++) {
        num = *iter * num;
    }
    return num;
}

xfer_prop tiramisu::communicator::get_xfer_props() const
{
    return prop;
}

std::vector<communicator *> tiramisu::communicator::collapse(int level, tiramisu::expr collapse_from_iter,
                                                             tiramisu::expr collapse_until_iter,
                                                             tiramisu::expr num_collapsed)
{

    std::vector<communicator *> ret;
    if (collapse_until_iter.get_expr_type() == tiramisu::e_val && collapse_until_iter.get_int32_value() == -1) {
        this->add_dim(num_collapsed);
        // Instead of fully removing the loop, we modify the collapsed loop to only have a single iteration.
        full_loop_level_collapse(level, collapse_from_iter);
    } else {
        std::vector<communicator *> comms =
                partial_loop_level_collapse<communicator>(level, collapse_from_iter, collapse_until_iter,
                                                          num_collapsed);
        ret.push_back(comms[0]);
        ret.push_back(comms[1]);
    }

    return ret;
}

std::string create_send_func_name(const xfer_prop chan)
{
    if (chan.contains_attr(MPI)) {
        std::string name = "tiramisu_MPI";
        if (chan.contains_attr(SYNC) && chan.contains_attr(BLOCK)) {
            name += "_Ssend";
        } else if (chan.contains_attr(SYNC) && chan.contains_attr(NONBLOCK)) {
            name += "_Issend";
        } else if (chan.contains_attr(ASYNC) && chan.contains_attr(BLOCK)) {
            name += "_Send";
        } else if (chan.contains_attr(ASYNC) && chan.contains_attr(NONBLOCK)) {
            name += "_Isend";
        }
        switch (chan.get_dtype()) {
            case p_uint8:
                name += "_uint8";
                break;
            case p_uint16:
                name += "_uint16";
                break;
            case p_uint32:
                name += "_uint32";
                break;
            case p_uint64:
                name += "_uint64";
                break;
            case p_int8:
                name += "_int8";
                break;
            case p_int16:
                name += "_int16";
                break;
            case p_int32:
                name += "_int32";
                break;
            case p_int64:
                name += "_int64";
                break;
            case p_float32:
                name += "_f32";
                break;
            case p_float64:
                name += "_f64";
                break;
        }
        return name;
    } else if (chan.contains_attr(CUDA)) {
        std::string name = "tiramisu_cudad_memcpy";
        if (chan.contains_attr(ASYNC)) {
            name += "_async";
        }
        if (chan.contains_attr(CPU2CPU)) {
            name += "_h2h";
        } else if (chan.contains_attr(CPU2GPU)) {
            name += "_h2d";
        } else if (chan.contains_attr(GPU2GPU)) {
            name += "_d2d";
        } else if (chan.contains_attr(GPU2CPU)) {
            name += "_d2h";
        } else {
            assert(false && "Unknown CUDA transfer direction");
        }

        return name;
    }
    assert(false && "Communication must be either MPI or CUDA!");
    return "";
}

int send::next_msg_tag = 0;

tiramisu::send::send(std::string iteration_domain_str, tiramisu::computation *producer, tiramisu::expr rhs,
                     xfer_prop prop, bool schedule_this, std::vector<expr> dims, tiramisu::function *fct) :
        communicator(iteration_domain_str, rhs, schedule_this, prop.get_dtype(),
                     prop, fct), producer(producer),
        msg_tag(tiramisu::expr(next_msg_tag++))
{
    _is_library_call = true;
    library_call_name = create_send_func_name(prop);
    expr mod_rhs(tiramisu::o_address_of, rhs.get_name(), rhs.get_access(), rhs.get_data_type());
    set_expression(mod_rhs);
}

tiramisu::expr tiramisu::send::get_msg_tag() const
{
    return msg_tag;
}

tiramisu::computation *tiramisu::send::get_producer() const
{
    return producer;
}

tiramisu::recv *tiramisu::send::get_matching_recv() const
{
    return matching_recv;
}

void tiramisu::send::set_matching_recv(tiramisu::recv *matching_recv)
{
    this->matching_recv = matching_recv;
}

bool tiramisu::send::is_send() const
{
    return true;
}

void tiramisu::send::add_definitions(std::string iteration_domain_str,
                                     tiramisu::expr e,
                                     bool schedule_this_computation, tiramisu::primitive_t t,
                                     tiramisu::function *fct)
{
    tiramisu::send *new_c = new tiramisu::send(iteration_domain_str, this->producer, e, this->prop,
                                               schedule_this_computation, {}, fct);
    new_c->set_matching_recv(this->get_matching_recv());
    new_c->set_src(this->get_src());
    new_c->is_first = false;
    new_c->first_definition = this;
    this->updates.push_back(new_c);
}

tiramisu::expr tiramisu::send::get_src() const
{
    return src;
}

tiramisu::expr tiramisu::send::get_dest() const
{
    return dest;
}

void tiramisu::send::set_src(tiramisu::expr src)
{
    this->src = src;
}

void tiramisu::send::set_dest(tiramisu::expr dest)
{
    this->dest = dest;
}

void tiramisu::send::override_msg_tag(tiramisu::expr msg_tag)
{
    this->msg_tag = msg_tag;
}

std::string create_recv_func_name(const xfer_prop chan)
{

    if (chan.contains_attr(MPI)) {
        std::string name = "tiramisu_MPI";
        if (chan.contains_attr(BLOCK)) {
            name += "_Recv";
        } else if (chan.contains_attr(NONBLOCK)) {
            name += "_Irecv";
        }
        switch (chan.get_dtype()) {
            case p_uint8:
                name += "_uint8";
                break;
            case p_uint16:
                name += "_uint16";
                break;
            case p_uint32:
                name += "_uint32";
                break;
            case p_uint64:
                name += "_uint64";
                break;
            case p_int8:
                name += "_int8";
                break;
            case p_int16:
                name += "_int16";
                break;
            case p_int32:
                name += "_int32";
                break;
            case p_int64:
                name += "_int64";
                break;
            case p_float32:
                name += "_f32";
                break;
            case p_float64:
                name += "_f64";
                break;
        }
        return name;
    } else {
        assert(false);
        return "";
    }
}

tiramisu::recv::recv(std::string iteration_domain_str, bool schedule_this, tiramisu::xfer_prop prop,
                     tiramisu::function *fct) : communicator(iteration_domain_str, tiramisu::expr(),
                                                             schedule_this, prop.get_dtype(), prop, fct)
{
    _is_library_call = true;
}

send * tiramisu::recv::get_matching_send() const
{
    return matching_send;
}

void tiramisu::recv::set_matching_send(send *matching_send)
{
    this->matching_send = matching_send;
    library_call_name = create_recv_func_name(prop);
}

bool tiramisu::recv::is_recv() const
{
    return true;
}

tiramisu::expr tiramisu::recv::get_src() const
{
    return src;
}

tiramisu::expr tiramisu::recv::get_dest() const
{
    return dest;
}

void tiramisu::recv::set_src(tiramisu::expr src)
{
    this->src = src;
}

void tiramisu::recv::set_dest(tiramisu::expr dest)
{
    this->dest = dest;
}

void tiramisu::recv::override_msg_tag(tiramisu::expr msg_tag)
{
    this->msg_tag = msg_tag;
}

tiramisu::expr tiramisu::recv::get_msg_tag() const
{
    return this->msg_tag;
}

void tiramisu::recv::add_definitions(std::string iteration_domain_str,
                                     tiramisu::expr e,
                                     bool schedule_this_computation, tiramisu::primitive_t t,
                                     tiramisu::function *fct)
{
    tiramisu::recv *new_c = new tiramisu::recv(iteration_domain_str, schedule_this_computation, this->prop, fct);
    new_c->set_matching_send(this->get_matching_send());
    new_c->set_dest(this->get_dest());
    new_c->is_first = false;
    new_c->first_definition = this;
    new_c->is_let = this->is_let;
    new_c->definition_ID = this->definitions_number;
    this->definitions_number++;

    if (new_c->get_expr().is_equal(this->get_expr()))
    {
        // Copy the associated let statements to the new definition.
        new_c->associated_let_stmts = this->associated_let_stmts;
    }

    this->updates.push_back(new_c);
}

tiramisu::send_recv::send_recv(std::string iteration_domain_str, tiramisu::computation *producer,
                               tiramisu::expr rhs, xfer_prop prop, bool schedule_this_computation,
                               std::vector<expr> dims, tiramisu::function *fct) :
        communicator(iteration_domain_str, rhs, schedule_this_computation, prop.get_dtype(), prop, fct),
        producer(producer)
{
    _is_library_call = true;
    library_call_name = create_send_func_name(prop);
    if (prop.contains_attr(CPU2GPU)) {
        expr mod_rhs(tiramisu::o_address_of, rhs.get_name(), rhs.get_access(), rhs.get_data_type());
        set_expression(mod_rhs);
    } else if (prop.contains_attr(GPU2CPU)) {
        // we will modify this again later
        expr mod_rhs(tiramisu::o_buffer, rhs.get_name(), rhs.get_access(), rhs.get_data_type());
        set_expression(mod_rhs);
    }
}

bool tiramisu::send_recv::is_send_recv() const
{
    return true;
}

tiramisu::wait::wait(tiramisu::expr rhs, xfer_prop prop, tiramisu::function *fct)
        : communicator(), rhs(rhs) {
    assert(rhs.get_op_type() == tiramisu::o_access && "The RHS expression for a wait should be an access!");
    tiramisu::computation *op = fct->get_computation_by_name(rhs.get_name())[0];
    isl_set *dom = isl_set_copy(op->get_iteration_domain());
    std::string new_name = std::string(isl_set_get_tuple_name(dom)) + "_wait";
    dom = isl_set_set_tuple_name(dom, new_name.c_str());
    init_computation(isl_set_to_str(dom), fct, rhs, true, tiramisu::p_async);
    _is_library_call = true;
    this->prop = prop;
}

tiramisu::wait::wait(std::string iteration_domain_str, tiramisu::expr rhs, xfer_prop prop,
                     bool schedule_this,
                     tiramisu::function *fct) : communicator(iteration_domain_str, rhs, schedule_this, tiramisu::p_async, fct), rhs(rhs) {
    _is_library_call = true;
    this->prop = prop;
    computation *comp = fct->get_computation_by_name(rhs.get_name())[0];
    comp->_is_nonblock_or_async = true;
}

std::vector<tiramisu::computation *> tiramisu::wait::get_op_to_wait_on() const {
    std::string op_name = this->get_expr().get_name();
    return this->get_function()->get_computation_by_name(op_name);
}

bool tiramisu::wait::is_wait() const
{
    return true;
}

void tiramisu::wait::add_definitions(std::string iteration_domain_str,
                                     tiramisu::expr e, bool schedule_this_computation, tiramisu::primitive_t t,
                                     tiramisu::function *fct)
{
    tiramisu::computation *new_c = new tiramisu::wait(iteration_domain_str, e, this->prop, schedule_this_computation,
                                                      fct);
    new_c->is_first = false;
    new_c->first_definition = this;
    this->updates.push_back(new_c);
}

void tiramisu::computation::full_loop_level_collapse(int level, tiramisu::expr collapse_from_iter)
{
    std::string collapse_from_iter_repr = "";
    if (global::get_loop_iterator_data_type() == p_int32) {
        collapse_from_iter_repr = collapse_from_iter.get_expr_type() == tiramisu::e_val ?
                                  std::to_string(collapse_from_iter.get_int32_value()) : collapse_from_iter.get_name();
    } else {
        collapse_from_iter_repr = collapse_from_iter.get_expr_type() == tiramisu::e_val ?
                                  std::to_string(collapse_from_iter.get_int64_value()) : collapse_from_iter.get_name();
    }
    isl_map *sched = this->get_schedule();
    isl_map *sched_copy = isl_map_copy(sched);
    int dim = loop_level_into_dynamic_dimension(level);
    const char *_dim_name = isl_map_get_dim_name(sched, isl_dim_out, dim);
    std::string dim_name = "";
    if (!_dim_name) { // Since dim names are optional...
        dim_name = "jr" + std::to_string(next_dim_name++);
        sched = isl_map_set_dim_name(sched, isl_dim_out, dim, dim_name.c_str());
    } else {
        dim_name = _dim_name;
    }
    std::string subtract_cst =
            dim_name + " > " + collapse_from_iter_repr; // > because you want a single iteration (iter 0)
    isl_map *ident = isl_set_identity(isl_set_copy(this->get_iteration_domain()));
    ident = isl_map_apply_domain(isl_map_copy(this->get_schedule()), ident);
    assert(isl_map_n_out(ident) == isl_map_n_out(sched));
    ident = isl_map_set_dim_name(ident, isl_dim_out, dim, dim_name.c_str());
    isl_map *universe = isl_map_universe(isl_map_get_space(ident));
    std::string transform_str = isl_map_to_str(universe);
    std::vector<std::string> parts;
    split_string(transform_str, "}", parts);
    transform_str = parts[0] + ": " + subtract_cst + "}";
    isl_map *transform = isl_map_read_from_str(this->get_ctx(), transform_str.c_str());
    if (collapse_from_iter.get_expr_type() != tiramisu::e_val) { // This might be a free variable
        transform = isl_map_add_free_var(collapse_from_iter_repr, transform, this->get_ctx());
    }
    sched = isl_map_subtract(sched, transform);
    // update all the dim names because they get removed in the transform
    assert(isl_map_n_out(sched) == isl_map_n_out(sched_copy)); // Shouldn't have added or removed any dims here...
    this->set_schedule(sched);
}

xfer tiramisu::computation::create_xfer(std::string send_iter_domain, std::string recv_iter_domain,
                                        tiramisu::expr send_dest, tiramisu::expr recv_src,
                                        xfer_prop send_prop, xfer_prop recv_prop,
                                        tiramisu::expr send_expr, tiramisu::function *fct) {
    if (send_prop.contains_attr(MPI)) {
        assert(recv_prop.contains_attr(MPI));
    } else if (send_prop.contains_attr(CUDA)) {
        assert(recv_prop.contains_attr(CUDA));
    }

    assert(send_expr.get_op_type() == tiramisu::o_access);
    tiramisu::computation *producer = fct->get_computation_by_name(send_expr.get_name())[0];

    isl_set *s_iter_domain = isl_set_read_from_str(producer->get_ctx(), send_iter_domain.c_str());
    isl_set *r_iter_domain = isl_set_read_from_str(producer->get_ctx(), recv_iter_domain.c_str());
    tiramisu::send *s = new tiramisu::send(isl_set_to_str(s_iter_domain), producer, send_expr, send_prop, true,
                                           {1}, producer->get_function());
    tiramisu::recv *r = new tiramisu::recv(isl_set_to_str(r_iter_domain), true, recv_prop, fct);
    isl_map *send_sched = s->gen_identity_schedule_for_iteration_domain();
    isl_map *recv_sched = r->gen_identity_schedule_for_iteration_domain();

    s->set_src(expr());
    s->set_dest(send_dest);
    r->set_src(recv_src);
    r->set_dest(expr());

    s->set_schedule(send_sched);
    r->set_schedule(recv_sched);
    s->set_matching_recv(r);
    r->set_matching_send(s);

    tiramisu::xfer c;
    c.s = s;
    c.r = r;
    c.sr = nullptr;

    return c;
}

xfer tiramisu::computation::create_xfer(std::string iter_domain_str, xfer_prop prop, tiramisu::expr expr,
                                        tiramisu::function *fct) {
    assert(expr.get_op_type() == tiramisu::o_access);
    tiramisu::computation *producer = fct->get_computation_by_name(expr.get_name())[0];

    isl_set *iter_domain = isl_set_read_from_str(producer->get_ctx(), iter_domain_str.c_str());
    tiramisu::send_recv *sr = new tiramisu::send_recv(isl_set_to_str(iter_domain), producer, expr, prop, true, {1},
                                                      producer->get_function());
    isl_map *sched = sr->gen_identity_schedule_for_iteration_domain();

    sr->set_schedule(sched);

    tiramisu::xfer c;
    c.s = nullptr;
    c.r = nullptr;
    c.sr = sr;

    return c;
}

void tiramisu::function::lift_dist_comps() {
    for (std::vector<tiramisu::computation *>::iterator comp = body.begin(); comp != body.end(); comp++) {
        if ((*comp)->is_send() || (*comp)->is_recv() || (*comp)->is_wait() || (*comp)->is_send_recv()) {
            xfer_prop chan = static_cast<tiramisu::communicator *>(*comp)->get_xfer_props();
            if (chan.contains_attr(MPI)) {
                lift_mpi_comp(*comp);
            } else if (chan.contains_attr(CUDA)) {
                assert(false && "CUDA lifter not implemented yet");
//                lift_cuda_comp(*comp);
            } else {
                assert(false);
            }
        }
    }
}

void tiramisu::function::lift_mpi_comp(tiramisu::computation *comp) {
    if (comp->is_send()) {
        send *s = static_cast<send *>(comp);
        tiramisu::expr num_elements(s->get_num_elements());
        tiramisu::expr send_type(s->get_xfer_props().get_dtype());
        bool isnonblock = s->get_xfer_props().contains_attr(NONBLOCK);
        s->rhs_argument_idx = 3;
        s->library_call_args.resize(isnonblock ? 6 : 5);
        s->library_call_args[0] = tiramisu::expr(tiramisu::o_cast, p_int32, num_elements);
        s->library_call_args[1] = tiramisu::expr(tiramisu::o_cast, p_int32, s->get_dest());
        s->library_call_args[2] = tiramisu::expr(tiramisu::o_cast, p_int32, s->get_msg_tag());
        s->library_call_args[4] = send_type;
        if (isnonblock) {
            // This additional RHS argument is to the request buffer. It is really more of a side effect.
            s->wait_argument_idx = 5;
        }
    } else if (comp->is_recv()) {
        recv *r = static_cast<recv *>(comp);
        send *s = r->get_matching_send();
        tiramisu::expr num_elements(r->get_num_elements());
        tiramisu::expr recv_type(s->get_xfer_props().get_dtype());
        bool isnonblock = r->get_xfer_props().contains_attr(NONBLOCK);
        r->lhs_argument_idx = 3;
        r->library_call_args.resize(isnonblock ? 6 : 5);
        r->library_call_args[0] = tiramisu::expr(tiramisu::o_cast, p_int32, num_elements);
        r->library_call_args[1] = tiramisu::expr(tiramisu::o_cast, p_int32, r->get_src());
        r->library_call_args[2] = tiramisu::expr(tiramisu::o_cast, p_int32, r->get_msg_tag().is_defined() ?
                                                                            r->get_msg_tag() : s->get_msg_tag());
        r->library_call_args[4] = recv_type;
        r->lhs_access_type = tiramisu::o_address_of;
        if (isnonblock) {
            // This RHS argument is to the request buffer. It is really more of a side effect.
            r->wait_argument_idx = 5;
        }
    } else if (comp->is_wait()) {
        wait *w = static_cast<wait *>(comp);
        w->rhs_argument_idx = 0;
        w->library_call_args.resize(1);
        w->library_call_name = "tiramisu_MPI_Wait";
    }
}

void split_string(std::string str, std::string delimiter, std::vector<std::string> &vector)
{
    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        token = str.substr(0, pos);
        vector.push_back(token);
        str.erase(0, pos + delimiter.length());
    }
    token = str.substr(0, pos);
    vector.push_back(token);
}

isl_map *isl_map_add_free_var(const std::string &free_var_name, isl_map *map, isl_ctx *ctx) {
    isl_map *final_map = nullptr;

    // first, check to see if this variable is actually a free variable. If not, then we don't need to add it.
    int num_domain_dims = isl_map_dim(map, isl_dim_in);
    for (int i = 0; i < num_domain_dims; i++) {
        if (std::strcmp(isl_map_get_dim_name(map, isl_dim_in, i), free_var_name.c_str()) == 0) {
            return map;
        }
    }
    int num_range_dims = isl_map_dim(map, isl_dim_out);
    for (int i = 0; i < num_range_dims; i++) {
        if (std::strcmp(isl_map_get_dim_name(map, isl_dim_out, i), free_var_name.c_str()) == 0) {
            return map;
        }
    }

    std::string map_str = isl_map_to_str(map);
    std::vector<std::string> parts;
    split_string(map_str, "{", parts);
    if (parts[0] != "") { // A free variable already exists, so add this variable to that box
        std::vector<std::string> free_parts;
        split_string(parts[0], "[", free_parts);
        // remove the right bracket
        std::vector<std::string> tmp;
        split_string(free_parts[free_parts.size() - 1], "]", tmp);
        free_parts.insert(free_parts.end(), tmp[0]);
        std::string free_vars = "";
        int ctr = 0;
        for (auto s: free_parts) {
            if (s == free_var_name) {
                // The variable name was already in the box, so we don't actually need to do anything
                return map;
            }
            free_vars += ctr++ == 0 ? s : "," + s;
        }
        free_vars += "," + free_var_name;
        free_vars = "[" + free_vars + "]" + "{" + parts[1];
        final_map = isl_map_read_from_str(ctx, free_vars.c_str());
    } else {
        std::string m = "[" + free_var_name + "]->{" + parts[1];
        final_map = isl_map_read_from_str(ctx, m.c_str());
    }
    assert(final_map && "Adding free param to map resulted in a null isl_map");
    return final_map;
}


expr tiramisu::computation::get_span(int level)
{
    this->check_dimensions_validity({level});
    tiramisu::expr loop_upper_bound =
            tiramisu::utility::get_bound(this->get_trimmed_time_processor_domain(),
                                         level, true);

    tiramisu::expr loop_lower_bound =
            tiramisu::utility::get_bound(this->get_trimmed_time_processor_domain(),
                                         level, false);

    tiramisu::expr loop_bound = loop_upper_bound - loop_lower_bound + value_cast(global::get_loop_iterator_data_type(), 1);
    return loop_bound.simplify();
}

void tiramisu::buffer::tag_gpu_shared() {
    location = cuda_ast::memory_location::shared;
    set_auto_allocate(false);
}

void tiramisu::buffer::tag_gpu_constant() {
    location = cuda_ast::memory_location::constant;
}

void tiramisu::buffer::tag_gpu_global() {
    location = cuda_ast::memory_location::global;
}

void tiramisu::buffer::tag_gpu_register() {
    bool is_single_val = this->get_n_dims() == 1 && this->get_dim_sizes()[0].get_expr_type() == e_val && this->get_dim_sizes()[0].get_int_val() == 1;
    assert(is_single_val && "Buffer needs to correspond to a single value to be in register");
    location = cuda_ast::memory_location::reg;
    set_auto_allocate(false);
}
}
