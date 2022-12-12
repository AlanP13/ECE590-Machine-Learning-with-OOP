#include <assert.h>

#include "operators.h"

eval_op_prototypes &eval_op_prototypes::instance()
{
    static eval_op_prototypes instance; // the only instance
    return instance;
}

std::shared_ptr<eval_op> eval_op_prototypes::locate(
    std::string name)
{
    auto itr = proto_map_.find(name);
    assert(itr != proto_map_.end());
    return itr->second;
}

void eval_const::store_prototype(
    eval_op_proto_map &proto_map)
{
    assert(proto_map.find("Const") == proto_map.end());
    proto_map["Const"] = std::make_shared<eval_const>();
}

void eval_input::store_prototype(
    eval_op_proto_map &proto_map)
{
    assert(proto_map.find("Input") == proto_map.end());
    proto_map["Input"] = std::make_shared<eval_input>();
}

void eval_add::store_prototype(
    eval_op_proto_map &proto_map)
{
    assert(proto_map.find("Add") == proto_map.end());
    proto_map["Add"] = std::make_shared<eval_add>();
}

void eval_sub::store_prototype(
    eval_op_proto_map &proto_map)
{
    assert(proto_map.find("Sub") == proto_map.end());
    proto_map["Sub"] = std::make_shared<eval_sub>();
}

void eval_mul::store_prototype(
    eval_op_proto_map &proto_map)
{
    assert(proto_map.find("Mul") == proto_map.end());
    proto_map["Mul"] = std::make_shared<eval_mul>();
}

void eval_neg::store_prototype(
    eval_op_proto_map &proto_map)
{
    assert(proto_map.find("Neg") == proto_map.end());
    proto_map["Neg"] = std::make_shared<eval_neg>();
}

std::shared_ptr<eval_op> eval_const::clone(
    const expression &expr)
{
    return std::make_shared<eval_const>(expr);
}

std::shared_ptr<eval_op> eval_input::clone(
    const expression &expr)
{
    return std::make_shared<eval_input>(expr);
}

std::shared_ptr<eval_op> eval_add::clone(
    const expression &expr)
{
    return std::make_shared<eval_add>(expr);
}

std::shared_ptr<eval_op> eval_sub::clone(
    const expression &expr)
{
    return std::make_shared<eval_sub>(expr);
}

std::shared_ptr<eval_op> eval_mul::clone(
    const expression &expr)
{
    return std::make_shared<eval_mul>(expr);
}

std::shared_ptr<eval_op> eval_neg::clone(
    const expression &expr)
{
    return std::make_shared<eval_neg>(expr);
}

eval_op::eval_op()
    : expr_id_(-1)
{
}

eval_unary::eval_unary()
{
}

eval_binary::eval_binary()
{
}

eval_const::eval_const()
{
}

eval_input::eval_input()
{
}

eval_add::eval_add()
{
}

eval_sub::eval_sub()
{
}

eval_mul::eval_mul()
{
}

eval_neg::eval_neg()
{
}

eval_op::eval_op(const expression &expr)
    : expr_id_(expr.get_expr_id()),
    op_name_(expr.get_op_name()),
    op_type_(expr.get_op_type()),
    inputs_(expr.get_inputs())
{
}

eval_unary::eval_unary(const expression &expr)
    : eval_op(expr)
{
}

eval_binary::eval_binary(const expression &expr)
    : eval_op(expr)
{
}

eval_const::eval_const(const expression &expr)
    : eval_op(expr), value_(expr.get_op_param_ndarray("value"))
{
}

eval_input::eval_input(const expression &expr)
    : eval_op(expr)
{
}

eval_add::eval_add(const expression &expr)
    : eval_binary(expr)
{
}

eval_sub::eval_sub(const expression &expr)
    : eval_binary(expr)
{
}

eval_mul::eval_mul(const expression &expr)
    : eval_binary(expr)
{
}

eval_neg::eval_neg(const expression &expr)
    : eval_unary(expr)
{
}

eval_op::~eval_op()
{
}

eval_unary::~eval_unary()
{
}

eval_binary::~eval_binary()
{
}

void eval_op::eval(
    vars_type &variables,
    const kwargs_type &kwargs)
{
    assert(false); // should be provided by derived classes
}

void eval_unary::eval(
    vars_type &variables,
    const kwargs_type &kwargs)
{
    assert(inputs_.size() == 1);
    auto it = variables.find(inputs_[0]);
    assert(it != variables.end());
    
    variables[expr_id_] = compute(it->second);
}

void eval_binary::eval(
    vars_type &variables,
    const kwargs_type &kwargs)
{
    assert(inputs_.size() == 2);
    auto ita = variables.find(inputs_[0]);
    auto itb = variables.find(inputs_[1]);
    assert(ita != variables.end() && itb != variables.end());
    
    variables[expr_id_] = compute(ita->second, itb->second);
}

void eval_const::eval(
    vars_type &variables,
    const kwargs_type &kwargs)
{
    variables[expr_id_] = value_;
}

void eval_input::eval(
    vars_type &variables,
    const kwargs_type &kwargs)
{
    auto iter = kwargs.find(op_name_);
    assert(iter != kwargs.end());
    variables[expr_id_] = iter->second;
}

tensor eval_add::compute(
    const tensor &a,
    const tensor &b)
{
    return a.add(b);
}

tensor eval_sub::compute(
    const tensor &a,
    const tensor &b)
{
    return a.sub(b);
}

tensor eval_mul::compute(
    const tensor &a,
    const tensor &b)
{
    return a.mul(b);
}

tensor eval_neg::compute(
    const tensor &t)
{
    return t.neg();
}