#ifndef OPERATORS_H
#define OPERATORS_H

#include <memory>

#include "expression.h"
#include "tensor.h"

class eval_op;

typedef std::map<int, tensor> vars_type;
typedef std::map<std::string, tensor> kwargs_type;
typedef std::map<std::string, std::shared_ptr<eval_op>> eval_op_proto_map;

class eval_op_prototypes {
private:
    eval_op_proto_map proto_map_;

    eval_op_prototypes();
    // prevent creation of additional instances
    eval_op_prototypes(const eval_op_prototypes &)
        = delete;
public:
    static eval_op_prototypes &instance(); // access the only instance
    std::shared_ptr<eval_op> locate(std::string name);
}; // class eval_op_prototypes

class eval_op {
protected:
    int expr_id_;
    std::string op_name_, op_type_;
    std::vector<int> inputs_;

    eval_op();
    eval_op(const expression &expr);
    virtual ~eval_op();
public:
    virtual void eval(
        vars_type &variables,
        const kwargs_type &kwargs)
        = 0;
    
    virtual std::shared_ptr<eval_op> clone(
        const expression &expr)
        = 0;
}; // class eval_op

class eval_unary: public eval_op {
private:
    virtual tensor compute(const tensor &t)
        = 0;
protected:
    eval_unary();
    eval_unary(const expression &expr);
    virtual ~eval_unary();
public:
    void eval(
        vars_type &variables,
        const kwargs_type &kwargs)
        final;
}; // class eval_unary

class eval_binary: public eval_op {
private:
    virtual tensor compute(
        const tensor &a,
        const tensor &b)
        = 0;
protected:
    eval_binary();
    eval_binary(const expression &expr);
    virtual ~eval_binary();
public:
    void eval(
        vars_type &variables,
        const kwargs_type &kwargs)
        final;
}; // class eval_binary

class eval_const: public eval_op {
private:
    const tensor value_;
public:
    static void store_prototype(
        eval_op_proto_map &proto_map);
    
    eval_const();
    eval_const(const expression &expr);

    void eval(
        vars_type &variables,
        const kwargs_type &kwargs)
        override;
    
    std::shared_ptr<eval_op> clone(
        const expression &expr)
        override;
}; // class eval_const

class eval_input: public eval_op {
public:
    static void store_prototype(
        eval_op_proto_map &proto_map);
    
    eval_input();
    eval_input(const expression &expr);

    void eval(
        vars_type &variables,
        const kwargs_type &kwargs)
        override;
    
    std::shared_ptr<eval_op> clone(
        const expression &expr)
        override;
}; // class eval_input

class eval_add: public eval_binary {
private:
    tensor compute(
        const tensor &a,
        const tensor &b)
        override;
public:
    static void store_prototype(
        eval_op_proto_map &proto_map);
    
    eval_add();
    eval_add(const expression &expr);
    
    std::shared_ptr<eval_op> clone(
        const expression &expr)
        override;
}; // class eval_add

class eval_sub: public eval_binary {
private:
    tensor compute(
        const tensor &a,
        const tensor &b)
        override;
public:
    static void store_prototype(
        eval_op_proto_map &proto_map);
    
    eval_sub();
    eval_sub(const expression &expr);
    
    std::shared_ptr<eval_op> clone(
        const expression &expr)
        override;
}; // class eval_sub

class eval_mul: public eval_binary {
private:
    tensor compute(
        const tensor &a,
        const tensor &b)
        override;
public:
    static void store_prototype(
        eval_op_proto_map &proto_map);
    
    eval_mul();
    eval_mul(const expression &expr);
    
    std::shared_ptr<eval_op> clone(
        const expression &expr)
        override;
}; // class eval_mul

class eval_neg: public eval_unary {
private:
    tensor compute(const tensor &t) override;
public:
    static void store_prototype(
        eval_op_proto_map &proto_map);
    
    eval_neg();
    eval_neg(const expression &expr);
    
    std::shared_ptr<eval_op> clone(
        const expression &expr)
        override;
}; // class eval_neg

#endif // OPERATORS_H
