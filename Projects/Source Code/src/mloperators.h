#ifndef MLOPERATORS_H
#define MLOPERATORS_H

#include "operators.h"

class eval_ml: public eval_unary {
private:
    tensor compute(const tensor &t) override;
    virtual tensor compute(
        size_t dim,
        size_t d_size,
        const size_t *s,
        const double *d)
        = 0;
protected:
    eval_ml();
    eval_ml(const expression &expr);
    virtual ~eval_ml();
}; // class eval_ml

class eval_relu: public eval_ml {
private:
    tensor compute(
        size_t dim,
        size_t d_size,
        const size_t *s,
        const double *d)
        override;
public:
    static void store_prototype(
        eval_op_proto_map &proto_map);
    
    eval_relu();
    eval_relu(const expression &expr);
    
    std::shared_ptr<eval_op> clone(
        const expression &expr)
        override;
}; // class eval_relu

class eval_flatten: public eval_ml {
private:
    tensor compute(
        size_t dim,
        size_t d_size,
        const size_t *s,
        const double *d)
        override;
public:
    static void store_prototype(
        eval_op_proto_map &proto_map);
    
    eval_flatten();
    eval_flatten(const expression &expr);
    
    std::shared_ptr<eval_op> clone(
        const expression &expr)
        override;
}; // class eval_flatten

class eval_input2d: public eval_op {
private:
    const double height_;
    const double width_;
    const double in_channels_;
public:
    static void store_prototype(
        eval_op_proto_map &proto_map);
    
    eval_input2d();
    eval_input2d(const expression &expr);

    void eval(
        vars_type &variables,
        const kwargs_type &kwargs)
        override;
    
    std::shared_ptr<eval_op> clone(
        const expression &expr)
        override;
}; // class eval_input2d

class eval_linear: public eval_unary {
private:
    const tensor weight_;
    const tensor bias_;

    tensor compute(const tensor &t) override;
public:
    static void store_prototype(
        eval_op_proto_map &proto_map);
    
    eval_linear();
    eval_linear(const expression &expr);
    
    std::shared_ptr<eval_op> clone(
        const expression &expr)
        override;
}; // class eval_linear

class eval_maxpool2d: public eval_unary {
private:
    const int kernel_size_;
    const int stride_;

    tensor compute(const tensor &t) override;
public:
    static void store_prototype(
        eval_op_proto_map &proto_map);
    
    eval_maxpool2d();
    eval_maxpool2d(const expression &expr);
    
    std::shared_ptr<eval_op> clone(
        const expression &expr)
        override;
}; // class eval_maxpool2d

class eval_conv2d: public eval_unary {
private:
    const double in_channels_;
    const double out_channels_;
    const double kernel_size_;
    const tensor weight_;
    const tensor bias_;

    tensor compute(const tensor &t) override;
public:
    static void store_prototype(
        eval_op_proto_map &proto_map);
    
    eval_conv2d();
    eval_conv2d(const expression &expr);
    
    std::shared_ptr<eval_op> clone(
        const expression &expr)
        override;
}; // class eval_conv2d

#endif
