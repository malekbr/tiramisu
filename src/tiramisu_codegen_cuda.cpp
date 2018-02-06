//
// Created by malek on 12/15/17.
//

#include <isl/printer.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/constraint.h>
#include <isl/space.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <tiramisu/type.h>
#include <tiramisu/expr.h>

#include <string>
#include <tiramisu/cuda_ast.h>
#include <isl/ast_type.h>
#include <isl/ast.h>

namespace tiramisu
{


    cuda_ast::statement * tiramisu::cuda_ast::generator::cuda_stmt_from_isl_node(isl_ast_node *node) {
        isl_ast_node_type type = isl_ast_node_get_type(node);

        switch(type)
        {
            case isl_ast_node_for:
                return cuda_stmt_handle_isl_for(node);
            case isl_ast_node_block:
                return cuda_stmt_handle_isl_block(node);
            case isl_ast_node_if:
                return cuda_stmt_handle_isl_if(node);
            case isl_ast_node_mark:
            DEBUG(3, tiramisu::str_dump("mark"));
                return nullptr;
            case isl_ast_node_user:
                return cuda_stmt_handle_isl_user(node);
            default:
            DEBUG(3, tiramisu::str_dump("default"));
                return nullptr;
        }
    }

    cuda_ast::statement * tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_if(isl_ast_node *node) {
        isl_ast_expr * condition = isl_ast_node_if_get_cond(node);
        isl_ast_node * then_body = isl_ast_node_if_get_then(node);
        isl_ast_node * else_body = isl_ast_node_if_get_else(node);
        return new cuda_ast::if_condition{cuda_stmt_handle_isl_expr(condition, node),
                                          cuda_stmt_from_isl_node(then_body),
                                          cuda_stmt_from_isl_node(else_body)};

    }

    cuda_ast::statement * tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_block(isl_ast_node *node) {
        isl_ast_node_list * children_list = isl_ast_node_block_get_children(node);
        const int block_length = isl_ast_node_list_n_ast_node(children_list);
        auto * b = new block;
        for (int i = 0; i < block_length; i++)
        {
            isl_ast_node * child_node = isl_ast_node_list_get_ast_node(children_list, i);
            b->add_statement(cuda_stmt_from_isl_node(child_node));
        }
        return b;
    }

    void
    tiramisu::cuda_ast::generator::cuda_stmt_foreach_isl_expr_list(isl_ast_expr *node, const std::function<void(int, isl_ast_expr *)> &fn, int start) {
        int n = isl_ast_expr_get_op_n_arg(node);
        for(int i = start; i < n; i ++)
        {
            fn(i, isl_ast_expr_get_op_arg(node, i));
        }
    }

    cuda_ast::statement * tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_for(isl_ast_node *node) {
        isl_ast_expr * iterator = isl_ast_node_for_get_iterator(node);
        isl_id * iterator_id = isl_ast_expr_get_id(iterator);
        std::string iterator_name(isl_id_get_name(iterator_id));
        DEBUG(3, tiramisu::str_dump("The iterator name is: ", iterator_name.c_str()));

        isl_ast_expr * condition = isl_ast_node_for_get_cond(node);
        isl_ast_expr * incrementor = isl_ast_node_for_get_inc(node);
        isl_ast_expr * initializer = isl_ast_node_for_get_init(node);


        // TODO check if degenerate

        m_scalar_data.insert(
                std::make_pair(iterator_name,
                               std::make_pair(tiramisu::global::get_loop_iterator_default_data_type(),
                                              cuda_ast::memory_location::reg)));

        auto * it = (cuda_ast::scalar *)(cuda_stmt_handle_isl_expr(iterator, node));

        auto * result = new cuda_ast::for_loop{
                new declaration{new scalar_assignment{it, cuda_stmt_handle_isl_expr(initializer, node)}},
                cuda_stmt_handle_isl_expr(condition, node),
                new binary{it->get_type(), it, cuda_stmt_handle_isl_expr(incrementor, node), "+="},
                cuda_stmt_from_isl_node(isl_ast_node_for_get_body(node))};

        m_scalar_data.erase(iterator_name);

        return result;
    }

    cuda_ast::statement * tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_expr(isl_ast_expr *expr, isl_ast_node *node) {
        isl_ast_expr_type type = isl_ast_expr_get_type(expr);
        switch(type)
        {
            case isl_ast_expr_op:
                DEBUG(3, tiramisu::str_dump("isl op"));
                return cuda_stmt_handle_isl_op_expr(expr, node);
            case isl_ast_expr_id:
            {
                isl_id * id = isl_ast_expr_get_id(expr);
                std::string id_string(isl_id_get_name(id));
                DEBUG(3, std::cout << '"' << id_string << '"' );
                // TODO handle scheduled lets
                auto scalar_it = m_scalar_data.find(id_string);
                assert(scalar_it != m_scalar_data.end() && "Unknown name");
                return new cuda_ast::scalar{scalar_it->second.first, id_string, scalar_it->second.second};
            }
            case isl_ast_expr_int:
                return cuda_stmt_handle_isl_val(isl_ast_expr_get_val(expr));
            default:
                DEBUG(3, tiramisu::str_dump("expr default"));
                return nullptr;
                break;
        }
    }


    cuda_ast::value * tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_val(isl_val *node) {
        // TODO handle infinity
        long num = isl_val_get_num_si(node);
        long den = isl_val_get_den_si(node);
        assert(den == 1);
        return new cuda_ast::value{global::get_loop_iterator_default_data_type(), num};
    }


    cuda_ast::statement* tiramisu::cuda_ast::generator::parse_tiramisu(const tiramisu::expr &tiramisu_expr) {
        switch (tiramisu_expr.get_expr_type())
        {
            case e_val:
                return new cuda_ast::value{tiramisu_expr.get_data_type(), tiramisu_expr.get_int_val()};
            case e_var:
                return new cuda_ast::scalar{tiramisu_expr.get_data_type(), tiramisu_expr.get_name(), this->m_scalar_data[tiramisu_expr.get_name()].second};
            case e_none:
                assert(false);
            case e_op:
            {
                switch (tiramisu_expr.get_op_type())
                {
                    case o_access:
                    {
                        auto * b = this->get_buffer(tiramisu_expr.get_name());
                        std::vector<statement *> indices;
                        for (auto &access: tiramisu_expr.get_access())
                        {
                            indices.push_back(this->parse_tiramisu(access));
                        }
                        return new buffer_access{b, indices};
                    }
                    case o_call:
                    {
                        std::vector<statement *> operands{static_cast<size_t>(tiramisu_expr.get_n_arg())};
                        std::transform(tiramisu_expr.get_arguments().begin(), tiramisu_expr.get_arguments().end(), operands.begin(),
                                       std::bind(&generator::parse_tiramisu, this, std::placeholders::_1));
                        return new function_call{tiramisu_expr.get_data_type(), tiramisu_expr.get_name(), operands};
                    }
                    case o_cast:
                        return new cuda_ast::cast{tiramisu_expr.get_data_type(), parse_tiramisu(tiramisu_expr.get_operand(0))};

                    default: {
                        auto it = cuda_ast::tiramisu_operation_description.find(tiramisu_expr.get_op_type());
                        assert(it != cuda_ast::tiramisu_operation_description.cend());
                        const op_data_t & op_data = it->second;
                        std::vector<statement *> operands;
                        for (int i = 0; i < op_data.arity; i++)
                        {
                            operands.push_back(parse_tiramisu(tiramisu_expr.get_operand(i)));
                        }
                        if (op_data.infix)
                        {
                            assert(op_data.arity > 0 && op_data.arity < 4 && "Infix operators are either unary, binary, or tertiary.");
                            switch (op_data.arity)
                            {
                                case 1:
                                    return new cuda_ast::unary{tiramisu_expr.get_data_type(), operands[0], std::string{op_data.symbol}};
                                case 2:
                                    return new cuda_ast::binary{tiramisu_expr.get_data_type(), operands[0], operands[1], std::string{op_data.symbol}};
                                case 3:
                                    return new cuda_ast::ternary{tiramisu_expr.get_data_type(), operands[0], operands[1], operands[2], std::string{op_data.symbol}, std::string{op_data.next_symbol}};
                                default:
                                    assert(false && "Infix operators are either unary, binary, or tertiary.");
                            }
                        }
                        else
                        {
                            return new cuda_ast::function_call{tiramisu_expr.get_data_type(), op_data.symbol, operands};
                        }
                    }

                }
            }
        }
    }

    cuda_ast::buffer* tiramisu::cuda_ast::generator::get_buffer(const std::string &name) {
        auto it = m_buffers.find(name);
        if (it != m_buffers.end())
            return it->second;

        auto tiramisu_buffer = this->m_fct.get_buffers().at(name);
        std::vector<cuda_ast::statement *> sizes;
        for (auto &dim : tiramisu_buffer->get_dim_sizes())
        {

            sizes.push_back(this->parse_tiramisu(dim));
        }
        auto * buffer = new cuda_ast::buffer{tiramisu_buffer->get_elements_type(), tiramisu_buffer->get_name(), memory_location::global, sizes};
        m_buffers[name] = buffer;
        return buffer;
    }

    cuda_ast::statement * tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_op_expr(isl_ast_expr *expr,
                                                                                      isl_ast_node *node) {
        isl_ast_op_type op_type = isl_ast_expr_get_op_type(expr);
        if (op_type == isl_ast_op_call) {
            auto *comp = get_computation_annotated_in_a_node(node);
            auto result = comp->create_tiramisu_assignment();
            cuda_ast::buffer *b = this->get_buffer(result.first.get_name());
            return new buffer_assignment{b, parse_tiramisu(result.first.get_access()[0]), parse_tiramisu(result.second)};
        } else {
            auto it = isl_operation_description.find(op_type);
            assert(it != isl_operation_description.end() && "Operation not supported");
            auto &description = it->second;

            std::vector<statement *> operands;
            for (int i = 0; i < description.arity; i++)
            {
                operands.push_back(cuda_stmt_handle_isl_expr(isl_ast_expr_get_op_arg(expr, i), node));
            }
            primitive_t type = (description.type_preserving) ? operands.back()->get_type() : description.type; // Get the type of the last element because ternary condition

            if (description.infix)
            {
                switch (description.arity)
                {
                    case 1:
                        return new cuda_ast::unary{type, operands[0], std::string(description.symbol)};
                    case 2:
                        return new cuda_ast::binary{type, operands[0], operands[1], std::string(description.symbol)};
                    case 3:
                        return new cuda_ast::ternary{type, operands[0], operands[1], operands[2], std::string(description.symbol), std::string(description.next_symbol)};
                    default:
                        assert(false && "Infix operators are either unary, binary, or tertiary.");
                }
            } else
            {
                return new cuda_ast::function_call{type, description.symbol, operands};
            }
        }

    }

    cuda_ast::statement * tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_user(isl_ast_node *node) {
        isl_ast_expr * expr = isl_ast_node_user_get_expr(node);
        return cuda_stmt_handle_isl_expr(expr, node);
    }

    cuda_ast::assignment* tiramisu::cuda_ast::generator::cuda_generate_assignment(
            const std::pair<tiramisu::expr, tiramisu::expr> &assignment) {
        return nullptr; // TODO use or delete

    }

    void tiramisu::function::gen_cuda_stmt() {
        DEBUG_FCT_NAME(3);
        DEBUG_INDENT(4);

        DEBUG(3, this->gen_c_code());

        cuda_ast::generator generator{*this};

        std::cout << generator.cuda_stmt_from_isl_node(this->get_isl_ast())->print() << std::endl;

        DEBUG_INDENT(-4);

    }

    tiramisu::cuda_ast::generator::generator(tiramisu::function &fct) : m_fct(fct)
    {
        for (const tiramisu::constant & invariant : fct.get_invariants())
        {
            m_scalar_data.insert(std::make_pair(invariant.get_name(),
                                     std::make_pair(invariant.get_data_type(), cuda_ast::memory_location::constant)));
        }
    }


    cuda_ast::statement::statement(primitive_t type) : type(type) {}

    cuda_ast::cast::cast(primitive_t type, statement *stmt) : statement(type), to_be_cast(stmt) {}

    cuda_ast::abstract_identifier::abstract_identifier(primitive_t type, const std::string &name,
                                                       cuda_ast::memory_location location) : statement(type),
                                                                                             name(name),
                                                                                             location(location) {}

    const std::string &cuda_ast::abstract_identifier::get_name() const {
        return name;
    }

    cuda_ast::memory_location cuda_ast::abstract_identifier::get_location() const {
        return location;
    }

    cuda_ast::buffer::buffer(primitive_t type, const std::string &name, cuda_ast::memory_location location,
                             const std::vector<cuda_ast::statement *> &size) : abstract_identifier(type, name,
                                                                                                   location),
                                                                               size(size) {}

    cuda_ast::scalar::scalar(primitive_t type, const std::string &name, cuda_ast::memory_location location)
            : abstract_identifier(type, name, location) {}

    cuda_ast::value::value(primitive_t type, long val) : statement(type), val(val) {}

    cuda_ast::function_call::function_call(primitive_t type, const std::string &name,
                                           const std::vector<cuda_ast::statement *> &arguments) : statement(type),
                                                                                                  name(name), arguments(
                    arguments) {}

    cuda_ast::for_loop::for_loop(statement *initialization, cuda_ast::statement *condition,
                                 cuda_ast::statement *incrementer, statement *body) : initial_value(initialization),
                                                                                      condition(condition),
                                                                                      incrementer(incrementer),
                                                                                      body(body),
                                                                                      statement(p_none){}

    cuda_ast::block::block(): statement(p_none) {}

    cuda_ast::if_condition::if_condition(cuda_ast::statement *condition, statement *then_body,
                                         statement *else_body) : condition(condition), then_body(then_body),
                                                                 else_body(else_body), statement(p_none) {}

    void cuda_ast::block::add_statement(statement *stmt) {
        elements.push_back(stmt);
    }

    cuda_ast::buffer_access::buffer_access(cuda_ast::buffer *accessed,
                                           const std::vector<cuda_ast::statement *> &access) : statement(accessed->get_type()),
                                                                                               accessed(accessed),
                                                                                               access(access) {}

    cuda_ast::op::op(primitive_t type, const std::vector<statement *> &operands) : statement(type), m_operands(operands){}

    cuda_ast::unary::unary(primitive_t type, statement *operand, std::string &&op_symbol) : op(type, {operand}), m_op_symbol(op_symbol) {}

    cuda_ast::binary::binary(primitive_t type, statement *operand_1, statement *operand_2, std::string &&op_symbol)
            : op(type, {operand_1, operand_2}), m_op_symbol(op_symbol){}

    cuda_ast::ternary::ternary(primitive_t type, statement *operand_1, statement *operand_2, statement *operand_3,
                               std::string &&op_symbol_1, std::string &&op_symbol_2) : op(type, {operand_1, operand_2, operand_3}), m_op_symbol_1(op_symbol_1), m_op_symbol_2(op_symbol_2) {}


    cuda_ast::declaration::declaration(assignment *asgmnt) : statement(p_none), is_initialized(true) {content.asgmnt = asgmnt;}
    cuda_ast::declaration::declaration(abstract_identifier *identifier) : statement(p_none), is_initialized(true) {content.id = identifier;}

    primitive_t cuda_ast::statement::get_type() const {
        return type;
    }

    cuda_ast::assignment::assignment(primitive_t type) : cuda_ast::statement(type) {}

    cuda_ast::buffer_assignment::buffer_assignment(cuda_ast::buffer *buffer, statement *index_access, statement *rhs) : assignment(buffer->get_type()), m_buffer(buffer), m_index_access(index_access), m_rhs(rhs) {}
    cuda_ast::scalar_assignment::scalar_assignment(cuda_ast::scalar *scalar, statement *rhs) : assignment(scalar->get_type()), m_scalar(scalar), m_rhs(rhs) {}

    bool cuda_ast::op_data_t::operator==(const cuda_ast::op_data_t &rhs) const {
        return infix == rhs.infix &&
               arity == rhs.arity &&
               symbol == rhs.symbol &&
               next_symbol == rhs.next_symbol;
    }

    bool cuda_ast::op_data_t::operator!=(const cuda_ast::op_data_t &rhs) const {
        return !(rhs == *this);
    }

    std::string cuda_ast::statement::print() {
        std::stringstream ss;
        print(ss, "");
        return ss.str();
    }

    void cuda_ast::block::print(std::stringstream &ss, const std::string & base) {
        ss << "{\n";
        std::string new_base = base + "\t";
        for (auto &e : elements)
        {
            ss << new_base;
            e->print(ss, new_base);
            ss << ";\n";
        }
        ss << base << "}";
    }

    void cuda_ast::scalar::print(std::stringstream &ss, const std::string &base) {
        ss << get_name();
    }

    void cuda_ast::value::print(std::stringstream &ss, const std::string &base) {
        ss << val;
    }

    void cuda_ast::scalar_assignment::print(std::stringstream &ss, const std::string &base) {
        ss << m_scalar->get_name() << " = ";
        m_rhs->print(ss, base);
    }

    void cuda_ast::buffer_assignment::print(std::stringstream &ss, const std::string &base) {
        ss << m_buffer->get_name() << "[";
        m_index_access->print(ss, base);
        ss << "] = ";
        m_rhs->print(ss, base);
    }

    void cuda_ast::function_call::print(std::stringstream &ss, const std::string &base) {
        ss << name << "(";
        int i = 0;
        while (i < arguments.size())
        {
            arguments[i]->print(ss, base);
            i++;
            if (i < arguments.size())
            {
                ss << ", ";
            }
        }
        ss << ")";
    }

    void cuda_ast::for_loop::print(std::stringstream &ss, const std::string &base) {
        ss << "for (";
        initial_value->print(ss, base);
        ss << "; ";
        condition->print(ss, base);
        ss << "; ";
        incrementer->print(ss, base);
        ss << ")\n" << base;
        body->print(ss, base);
    }

    void cuda_ast::if_condition::print(std::stringstream &ss, const std::string &base) {
        ss << "if (";
        condition->print(ss, base);
        ss << ")\n" << base;
        then_body->print(ss, base);
        ss << "\n" << base << " else\n" << base;
        else_body->print(ss, base);
    }

    void cuda_ast::buffer_access::print(std::stringstream &ss, const std::string &base) {
        ss << accessed->get_name() << "[";
        int i = 0;
        while (i < access.size())
        {
            access[i]->print(ss, base);
            i++;
            if (i < access.size())
            {
                ss << ", ";
            }
        }
        ss << "]";
    }
    void cuda_ast::unary::print(std::stringstream &ss, const std::string &base) {
        ss << m_op_symbol;
        m_operands[0]->print(ss, base);
    }

    void cuda_ast::binary::print(std::stringstream &ss, const std::string &base) {
        m_operands[0]->print(ss, base);
        ss << " " << m_op_symbol << " ";
        m_operands[1]->print(ss, base);

    }

    void cuda_ast::ternary::print(std::stringstream &ss, const std::string &base) {
        m_operands[0]->print(ss, base);
        ss << " " << m_op_symbol_1 << " ";
        m_operands[1]->print(ss, base);
        ss << " " << m_op_symbol_2 << " ";
        m_operands[2]->print(ss, base);

    }

    void cuda_ast::buffer::print(std::stringstream &ss, const std::string &base) {
        ss << get_name();
    }

    void cuda_ast::declaration::print(std::stringstream &ss, const std::string &base) {
        if (is_initialized)
       {
            ss << tiramisu_type_to_cuda_type.at(content.asgmnt->get_type()) << " ";
            content.asgmnt->print(ss, base);
        }
        else
        {
            ss << tiramisu_type_to_cuda_type.at(content.id->get_type()) << " ";
            content.id->print(ss, base);
        }

    }

    void cuda_ast::cast::print(std::stringstream &ss, const std::string &base) {
        ss << "((" << tiramisu_type_to_cuda_type.at(get_type()) << ") ";
        to_be_cast->print(ss, base);
        ss << ")";
    }
};
