//
// Created by malek on 12/18/17.
//

#ifndef TIRAMISU_CUDA_AST_H
#define TIRAMISU_CUDA_AST_H

#define UNARY(op, x) {op, op_data_t{true, 1, (x)}}
#define UNARY_TYPED(op, x, T) {op, op_data_t{true, 2, (x), (T)}}
#define BINARY(op, x) {op, op_data_t{true, 2, (x)}}
#define BINARY_TYPED(op, x, T) {op, op_data_t{true, 2, (x), (T)}}
#define TERNARY(op, x, y) {op, op_data_t{true, 3, (x), (y)}}
#define TERNARY_TYPED(op, x, y, T) {op, op_data_t{true, 3, (x), (y), (T)}}
#define FN_CALL(op, x, n) {op, op_data_t{false, (n), (x)}}
#define FN_CALL_TYPED(op, x, n, T) {op, op_data_t{false, (n), (x), (T)}}

#include <tiramisu/type.h>
#include <string>
#include <vector>


namespace tiramisu
{
    class function;
namespace cuda_ast
{
    struct op_data_t
    {
        op_data_t(bool infix, int arity, std::string && symbol) : infix(infix), arity(arity), symbol(symbol) {}
        op_data_t(bool infix, int arity, std::string && symbol, std::string && next_symbol) : infix(infix), arity(arity), symbol(symbol), next_symbol(next_symbol) {}
        op_data_t(bool infix, int arity, std::string && symbol, primitive_t type) : infix(infix), arity(arity), symbol(symbol), type_preserving(
                false), type(type) {}
        op_data_t(bool infix, int arity, std::string && symbol, std::string && next_symbol, primitive_t type) : infix(infix), arity(arity), symbol(symbol), next_symbol(next_symbol), type_preserving(
                false), type(type) {}

        bool operator==(const op_data_t &rhs) const;

        bool operator!=(const op_data_t &rhs) const;

        bool infix;
        int arity;
        std::string symbol;
        std::string next_symbol = "";
        bool type_preserving = true;
        primitive_t type = p_none;
    };

    const std::unordered_map <tiramisu::op_t , op_data_t> tiramisu_operation_description = {
        UNARY(o_minus, "-"),
        FN_CALL(o_floor, "floor", 1),
        FN_CALL(o_sin, "sin", 1),
        FN_CALL(o_cos, "cos", 1),
        FN_CALL(o_tan, "tan", 1),
        FN_CALL(o_asin, "asin", 1),
        FN_CALL(o_acos, "acos", 1),
        FN_CALL(o_atan, "atan", 1),
        FN_CALL(o_abs, "abs", 1),
        FN_CALL(o_sqrt, "sqrt", 1),
        FN_CALL(o_expo, "exp", 1),
        FN_CALL(o_log, "log", 1),
        FN_CALL(o_ceil, "ceil", 1),
        FN_CALL(o_round, "round", 1),
        FN_CALL(o_trunc, "trunc", 1),
        BINARY(o_add, "+"),
        BINARY(o_sub, "-"),
        BINARY(o_mul, "*"),
        BINARY(o_div, "/"),
        BINARY(o_mod, "%"),
        BINARY(o_logical_and, "&&"),
        BINARY(o_logical_or, "||"),
        UNARY(o_logical_not, "!"),
        BINARY(o_eq, "=="),
        BINARY(o_ne, "!="),
        BINARY(o_le, "<="),
        BINARY(o_lt, "<"),
        BINARY(o_ge, ">="),
        BINARY(o_gt, ">"),
        FN_CALL(o_max, "max", 2),
        FN_CALL(o_max, "min", 2),
        BINARY(o_right_shift, ">>"),
        BINARY(o_left_shift, "<<"),
        TERNARY(o_select, "?", ":"),
        FN_CALL(o_lerp, "lerp", 3),
    };

    const std::unordered_map <isl_ast_op_type , op_data_t> isl_operation_description = {
        BINARY_TYPED(isl_ast_op_and, "&&", p_boolean),
        BINARY_TYPED(isl_ast_op_and_then, "&&", p_boolean),
        BINARY_TYPED(isl_ast_op_or, "||", p_boolean),
        BINARY_TYPED(isl_ast_op_or_else, "||", p_boolean),
        FN_CALL(isl_ast_op_max, "max", 2),
        FN_CALL(isl_ast_op_min, "min", 2),
        UNARY(isl_ast_op_minus, "-"),
        BINARY(isl_ast_op_add, "+"),
        BINARY(isl_ast_op_sub, "-"),
        BINARY(isl_ast_op_mul, "*"),
        BINARY(isl_ast_op_div, "/"),
        BINARY(isl_ast_op_fdiv_q, "/"),
        BINARY(isl_ast_op_pdiv_q, "/"),
        BINARY(isl_ast_op_pdiv_r, "%"),
        BINARY(isl_ast_op_zdiv_r, "%"),
        TERNARY(isl_ast_op_cond, "?", ":"),
        BINARY_TYPED(isl_ast_op_eq, "==", p_boolean),
        BINARY_TYPED(isl_ast_op_le, "<=", p_boolean),
        BINARY_TYPED(isl_ast_op_lt, "<", p_boolean),
        BINARY_TYPED(isl_ast_op_ge, ">=", p_boolean),
        BINARY_TYPED(isl_ast_op_gt, ">", p_boolean),
    };

    const std::unordered_map <tiramisu::primitive_t, std::string> tiramisu_type_to_cuda_type = {
            {p_none, "void"},
            {p_boolean, "bool"},
            {p_int8, "int8_t"},
            {p_uint8, "uint8_t"},
            {p_int16, "int16_t"},
            {p_uint16, "uint16_t"},
            {p_int32, "int32_t"},
            {p_uint32, "uint32_t"},
            {p_int64, "int64_t"},
            {p_uint64, "uint64_t"},
            {p_float32, "float"},
            {p_float64, "double"},
    };
enum class memory_location
{
    global,
    shared,
    constant,
    reg,
};

class abstract_node {

};

class statement : public abstract_node {
public:
    primitive_t get_type() const;
    std::string print();
    virtual void print(std::stringstream & ss, const std::string & base) = 0;

protected:
    explicit statement(primitive_t type);

private:
    tiramisu::primitive_t type;
};

class cast : public statement {
    statement * to_be_cast;
public:
    cast(primitive_t type, statement * stmt);
    void print(std::stringstream & ss, const std::string & base) override ;

};

class block : public statement {
private:
    std::vector<cuda_ast::statement *> elements;

public:
    void print(std::stringstream & ss, const std::string & base) override;

public:
    block();
    void add_statement(statement *);

};

class abstract_identifier : public statement
{
protected:
    abstract_identifier(primitive_t type, const std::string &name, memory_location location);

public:
    const std::string &get_name() const;
    memory_location get_location() const;


private:
    std::string name;
    cuda_ast::memory_location location;

public:

};

class buffer : public abstract_identifier
{
public:
    buffer(primitive_t type, const std::string &name, memory_location location, const std::vector<statement *> &size);
    void print(std::stringstream & ss, const std::string & base) override;


private:
    std::vector<cuda_ast::statement *> size;
};

class scalar : public abstract_identifier
{
public:
    scalar(primitive_t type, const std::string &name, memory_location location);

public:
    void print(std::stringstream & ss, const std::string & base) override;
};

class value : public statement
{
public:
    value(primitive_t type, long val);

public:
    void print(std::stringstream & ss, const std::string & base) override;

private:
    // TODO more generic
    long val;
};

class assignment : public statement
{
protected:
    explicit assignment(primitive_t type);
};

class scalar_assignment : public assignment
{
    cuda_ast::scalar * m_scalar;
    cuda_ast::statement * m_rhs;
public:
    scalar_assignment(cuda_ast::scalar * scalar, statement * rhs);
    void print(std::stringstream & ss, const std::string & base) override;

};

class buffer_assignment : public assignment
{
    cuda_ast::buffer * m_buffer;
    cuda_ast::statement * m_index_access;
    cuda_ast::statement * m_rhs;
public:
    void print(std::stringstream & ss, const std::string & base) override;
public:
    buffer_assignment(cuda_ast::buffer *buffer, statement *index_access, statement *rhs);
};

class function_call : public statement
{
public:
    function_call(primitive_t type, const std::string &name, const std::vector<statement *> &arguments);
public:
    void print(std::stringstream & ss, const std::string & base) override;

private:
    std::string name;
    std::vector<statement *> arguments;
};

class for_loop : public statement
{
public:
    for_loop(statement *initialization, statement *condition, statement *incrementer, statement *body);

public:
    void print(std::stringstream & ss, const std::string & base) override;

private:
    statement * initial_value;
    statement * condition;
    statement * incrementer;
    statement * body;
};

class if_condition : public statement
{
public:
    if_condition(statement *condition, statement *then_body, statement *else_body);

public:
    void print(std::stringstream & ss, const std::string & base) override;

private:
    statement * condition;
    statement * then_body;
    statement * else_body;
};

class buffer_access : public statement
{
public:
    buffer_access(buffer *accessed, const std::vector<statement *> &access);

public:
    void print(std::stringstream & ss, const std::string & base) override;

private:
    buffer * accessed;
    std::vector<cuda_ast::statement *> access;
};

class op : public statement
{

protected:
    op(primitive_t type, const std::vector<statement *> & operands);
    std::vector<statement *> m_operands;

};

class unary : public op
{
public:
    unary(primitive_t type, statement *operand, std::string &&op_symbol);
public:
    void print(std::stringstream & ss, const std::string & base) override;

private:
    std::string m_op_symbol;
};

class binary : public op
{
public:
    binary(primitive_t type, statement *operand_1, statement *operand_2, std::string &&op_symbol);

public:
    void print(std::stringstream & ss, const std::string & base) override;
private:
    std::string m_op_symbol;
};

class ternary : public op
{
public:
    ternary(primitive_t type, statement * operand_1, statement * operand_2, statement * operand_3, std::string &&op_symbol_1,  std::string &&op_symbol_2);

public:
    void print(std::stringstream & ss, const std::string & base) override;
private:
    std::string m_op_symbol_1;
    std::string m_op_symbol_2;

};

//class assignment : public statement
//{
//public:
//    assignment(primitive_t type, abstract_identifier *identifier, statement *value);
//
//private:
//    abstract_identifier * identifier;
//    statement * value;
//};

class declaration : public statement
{
public:
    explicit declaration (abstract_identifier * id);
    explicit declaration (assignment * asgmnt);
    void print(std::stringstream & ss, const std::string & base) override;


private:
    bool is_initialized;
    union {
        abstract_identifier * id;
        assignment * asgmnt;
    } content;
};

typedef std::unordered_map<std::string, std::pair<tiramisu::primitive_t, cuda_ast::memory_location> > scalar_data_t;

class generator
{
private:
    const tiramisu::function &m_fct;
    scalar_data_t m_scalar_data;
    std::unordered_map<std::string, cuda_ast::buffer *> m_buffers;
    cuda_ast::buffer * get_buffer(const std::string & name);
    cuda_ast::statement * parse_tiramisu(const tiramisu::expr & tiramisu_expr);
public:
    explicit generator(tiramisu::function &fct);

    statement * cuda_stmt_from_isl_node(isl_ast_node *node);
    statement * cuda_stmt_handle_isl_for(isl_ast_node *node);
    statement * cuda_stmt_handle_isl_block(isl_ast_node *node);
    statement * cuda_stmt_handle_isl_if(isl_ast_node *node);
    statement * cuda_stmt_handle_isl_user(isl_ast_node *node);
    cuda_ast::statement * cuda_stmt_handle_isl_expr(isl_ast_expr *expr, isl_ast_node *node);
    statement * cuda_stmt_handle_isl_op_expr(isl_ast_expr *expr, isl_ast_node *node);
    void cuda_stmt_foreach_isl_expr_list(isl_ast_expr *node, const std::function<void(int, isl_ast_expr *)> &fn, int start = 0);

    cuda_ast::assignment * cuda_generate_assignment(const std::pair<expr, expr> &assignment);
    static cuda_ast::value * cuda_stmt_handle_isl_val(isl_val *node);
};

}

}

#endif //TIRAMISU_CUDA_AST_H
