% how do I know if I have enough error responses for quantiles? check for
% that and only include the error quantiles for coherences where you have
% the enough error trials

function Loss = ACCtoBoundFit_quantiles_goLocked(Param, Target, ...
                                                 dt_ms, StdEv, ...
                                                 p_steps, go_step, TimOutPost, SimNum)

K     = Param(1);
Bound = Param(2);
NonDec= Param(3);

StimSet = Target.StimSet;
qLevels = Target.qLevels;

Pc_hat   = nan(numel(StimSet),1);
Qcorr_hat= nan(numel(StimSet), numel(qLevels));
Qerr_hat = nan(numel(StimSet), numel(qLevels));

T_total = go_step + TimOutPost;

parfor j = 1:numel(StimSet)
    mu = K * StimSet(j);

    % simulate evidence: drift during pulse only
    e = StdEv*sqrt(dt_ms) * randn(T_total, SimNum);
    e(1:p_steps,:) = e(1:p_steps,:) + mu;

    ACC = cumsum(e);

    % first bound hit time
    ACC_pos = ACC; ACC_pos(ACC>=Bound)=nan;
    mom_ev_pos = (ACC_pos + e) - ACC_pos;
    ACC_pos = cumsum(mom_ev_pos);
    RT_pos = sum(~isnan(ACC_pos),1) + 1;

    ACC_neg = ACC; ACC_neg(ACC<=-Bound)=nan;
    mom_ev_neg = (ACC_neg + e) - ACC_neg;
    ACC_neg = cumsum(mom_ev_neg);
    RT_neg = sum(~isnan(ACC_neg),1) + 1;

    RT_hit = RT_pos;
    m = RT_hit > RT_neg;
    RT_hit(m) = RT_neg(m);

    RT_hit(RT_hit > T_total | isnan(RT_hit)) = T_total;

    idx = sub2ind(size(ACC), RT_hit(:), (1:SimNum)');
    choiceUpper = ACC(idx) > 0;  % "upper choice" (not necessarily correct unless mapping known)

    % GO-locked RT (decision can happen before GO)
    rt_steps = RT_hit - go_step;
    rt_steps(rt_steps < 0) = 0;
    rt_sec = rt_steps * dt_ms/1000 + NonDec;

    % For fitting single pulse, we usually assume "upper choice" corresponds to correct
    % ONLY if stimuli are signed and you aligned them. If your Accuracy is already correct,
    % you should instead simulate signed stimuli or compare to Accuracy directly.
    % For now: treat Pcorr as P(upper) if that's your definition.
    Pc_hat(j) = mean(choiceUpper);

    % Quantiles conditional on "upper"/"lower" as proxy for correct/error
    rt_upper = rt_sec(choiceUpper==1);
    rt_lower = rt_sec(choiceUpper==0);

    % robust quantiles (avoid empty)
    if numel(rt_upper) >= 20
        Qcorr_hat(j,:) = quantile(rt_upper, qLevels);
    else
        Qcorr_hat(j,:) = nan;
    end
    if numel(rt_lower) >= 20
        Qerr_hat(j,:) = quantile(rt_lower, qLevels);
    else
        Qerr_hat(j,:) = nan;
    end
end

% ---- Loss construction ----
% 1) Choice-probability term
wp = 1;
Lp = mean((Pc_hat - Target.Pcorr).^2, 'omitnan');

% 2) Quantile terms (correct)
wq = 1;
Lq_corr = mean((Qcorr_hat - Target.Qcorr).^2, 'all', 'omitnan');

% 3) Error quantiles only where we have enough error data
maskErr = Target.useErr;
Lq_err = mean((Qerr_hat(maskErr,:) - Target.Qerr(maskErr,:)).^2, 'all', 'omitnan');

% Normalize by data spread so scales are comparable
Loss = wp * (sqrt(Lp) / std(Target.Pcorr)) ...
     + wq * (sqrt(Lq_corr) / std(Target.Qcorr(:), 'omitnan')) ...
     + wq * (sqrt(Lq_err)  / std(Target.Qerr(:),  'omitnan'));
end