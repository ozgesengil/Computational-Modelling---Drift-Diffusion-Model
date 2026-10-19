
% Same conception as ACCtoBoundFit2:
% - simulate diffusion with absorbing bounds
% - predict mean RT and mean choice per StimSet
% - compute normalized RMSE for perf and RT
%
% RT fix: convert steps -> seconds using dt consistently.

function Error = ACCtoBoundFit2_fixRT_goLocked(Param, Data, dt_ms, StdEv, StimSet, ...
                                              p_steps, go_step, TimOutPost, SimNum)
% GO-locked single-pulse fitting objective (keeps original conception: Gaussian increments + absorbing bounds)
% Param = [K Bound NonDec]
% Data  = [Perf RT] per StimSet level, RT is from GO in seconds

K     = Param(1);
Bound = Param(2);
NonDec= Param(3);

DataPerf = Data(:,1);
DataRT   = Data(:,2);

ModelPerf = zeros(length(StimSet),1);
ModelRT   = zeros(length(StimSet),1);

T_total = go_step + TimOutPost;  % simulate through GO and some post-GO time

parfor j=1:length(StimSet)
    mu = K * StimSet(j);

    % Evidence increments: drift only during the pulse, 0 drift after pulse
    e = StdEv*sqrt(dt_ms) * randn(T_total, SimNum);
    e(1:p_steps, :) = e(1:p_steps, :) + mu;

    ACC = cumsum(e);

    % Compute first bound hit time via min(upper, lower) using the same NaN trick
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

    % Clamp no-hit trials to end of simulation
    RT_hit(RT_hit > T_total | isnan(RT_hit)) = T_total;

    % Choice: sign of ACC at hit time (or at end if no hit)
    idx = sub2ind(size(ACC), RT_hit(:), (1:SimNum)');
    Choice = ACC(idx) > 0;

    % GO-locked RT:
    % if bound hit before GO -> RT_from_GO = 0
    % else RT_from_GO = (RT_hit - go_step) * dt_ms/1000
    rt_from_go_steps = RT_hit - go_step;
    rt_from_go_steps(rt_from_go_steps < 0) = 0;

    ModelRT(j,1)   = mean(rt_from_go_steps) * dt_ms/1000 + NonDec;
    ModelPerf(j,1) = mean(Choice);
end

ErrorPerf = sqrt(mean((ModelPerf - DataPerf).^2));
ErrorRT   = sqrt(mean((ModelRT   - DataRT  ).^2));
Error     = ErrorPerf/std(DataPerf) + ErrorRT/std(DataRT);
end