#ifndef EVALUATION_H
#define EVALUATION_H

#include "operators.h"
//#include "mloperators.h"

class evaluation
{
private:
    std::vector<std::shared_ptr<eval_op>> ops_;
    kwargs_type kwargs_;

    vars_type variables_;
    int result_;
public:
    evaluation(const std::vector<expression> &exprs);

    void add_kwargs_double(
        const char *key,
        double value);

    void add_kwargs_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);

    // return 0 for success
    int execute();

    // return the variable computed by the last expression
    tensor &get_result();
}; // class evaluation

#endif // EVALUATION_H
