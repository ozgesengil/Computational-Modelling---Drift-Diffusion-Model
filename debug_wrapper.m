function nll = debug_wrapper(Param, ExpData, cfg)

    persistent eval_count

    if isempty(eval_count)
        eval_count = 1;
    end

    nll = noshape_likelihood(Param, ExpData, cfg);

    fprintf('Eval %d | K = %.4f | Bound = %.4f | NonDec = %.4f | NLL = %.4f\n', ...
        eval_count, Param(1), Param(2), Param(3), nll);

    eval_count = eval_count + 1;

end