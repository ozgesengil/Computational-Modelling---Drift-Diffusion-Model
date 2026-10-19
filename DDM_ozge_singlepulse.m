%% 05.03.2026 - notes for the meeting

% General Note: I have realized I am using upper boundary reach as correct choice in all my codes whereas the stimuli is signed (left or right) - so half the simulated trials should be - and half be +. the logic is the same but I will correct that later.
% 
% 
% * initial investigation of parameters fitted for the single pulse:
%       K (evidence gain/drift scaling) (drift_rate_mu = K*coherence) : around 1. Therefore, drift rates are almost identical to coherence.
%       Bound is around 55 (quite high, meaning that subjects require a lot of accumulated evidence before committing. Large bounds produce higher accuracy, longer RT and smaller error rates.
%       NonDecTime is around 580 ms - ? should I have included the delay before the go signal for the response? Any drift/leak for that?
% 
%   * For RT - decreases with increased coherence. Higher drift - bound reached more quickly. the model reproduces that for single pulse.
%     but model underestimates RT reduction in high coherence ?
%   * For Performance - almost perfect accuracy for all coherences - could it be the bound is too high that it never crossed by mistake?
% * WHY? - could it be we used mean RT and mean performance in fitting error? Like to match the RT, bound is pushed, and then performance is ceiling?

%% Single-pulse fit -> use fitted params to simulate double-pulse (wait-to-end response)
close all; clear;

%% DATA LOAD
% You must provide single-pulse summary data as [Perf RT] per trial level.
% Perf in [0,1], RT in seconds. (matching the style of example coursework
% data.)
load('Data/ExpData.mat'); 

%% Build [Perf RT] summary from trial-level struct

% Choose the stimulus levels to fit (avoid 0 if you plan to plot log later)
StimSet = [0.016 0.032 0.064 0.128 0.256 0.512];
% to make sure the decimals are no problem when the equality of coherences
% are checked for
tol = 1e-12;

DataSingle = nan(numel(StimSet),2);

for j = 1:numel(StimSet)
    idx = abs(ExpData.StimStrength - StimSet(j)) < tol;
    %idx = (ExpData.StimStrength == StimSet(j));
    

    % Performance: mean accuracy at that strength
    DataSingle(j,1) = mean(ExpData.Acuracy(idx));

    % RT: mean RT at that strength (make sure it's in seconds)
    DataSingle(j,2) = mean(ExpData.RT(idx));
end

%% Optional sanity checks
assert(all(~isnan(DataSingle(:))), 'Some StimSet levels have no trials in your data.');

% timing
singlePulseDur_ms = 100;  % duration of single-pulse stimulus
pulseDur_ms       = 100;  % duration of each pulse in double-pulse
gapDur_ms         = 300;  % gap between pulses in double-pulse
goDelay_ms        = 50;   % latency between end of last pulse and GO screen (set to your true value)

% Optimization init guess: [K Bound NonDec]
Param0 = [0.35, 60, 0.35];

% Simulation fidelity / speed
SimNum_fit = 100;      % match your original code
SimNum_dp  = 400;      % more trials for smoother double-pulse predictions

% Plotting
doPlots = true;

%% FIXED MODEL SETTINGS 
% (match these to original conception)
StimSet = [0.016 0.032 0.064 0.128 0.256 0.512];
dt_ms      = 0.1;     % ms, time step used in your fitting function
StdEv   = 1;
TimOutPost  = 6000;    % max steps in single-pulse sim (as in your function)

% Convert double-pulse schedule to steps (dt is treated as "ms per step" here)
p_steps   = round(pulseDur_ms / dt_ms);
gap_steps = round(gapDur_ms   / dt_ms);
T_dp      = p_steps + gap_steps + p_steps;  % total steps until response is allowed

%% PREP DATA
DataPerf = DataSingle(:,1);
DataRT   = DataSingle(:,2);

assert(numel(DataPerf)==numel(StimSet) && numel(DataRT)==numel(StimSet), ...
    'DataSingle must be 6xn corresponding to StimSet.');

%% PRECOMPUTE GO STEP INDICES
p_steps_SP   = round(singlePulseDur_ms / dt_ms);
go_step_SP   = round((singlePulseDur_ms + goDelay_ms) / dt_ms); % GO time index in steps from trial start

p_steps_DP   = round(pulseDur_ms / dt_ms);
gap_steps_DP = round(gapDur_ms / dt_ms);
go_step_DP   = round((pulseDur_ms + gapDur_ms + pulseDur_ms + goDelay_ms) / dt_ms);

%% FIT SINGLE-PULSE PARAMETERS 
opts = optimset('MaxFunEvals', 800, 'MaxIter', 800, 'Display', 'iter');
%opts =
%optimset('MaxFunEvals',800,'MaxIter',800,'TolX',1e-4,'TolFun',1e-4); % if
%the fit is still unstable later, you can use this one instead

q0 = log(max(Param0, 1e-6));

obj = @(q) ACCtoBoundFit2_fixRT_goLocked(exp(q), DataSingle, ...
                                        dt_ms, StdEv, StimSet, ...
                                        p_steps_SP, go_step_SP, TimOutPost, SimNum_fit);


% for RT quantiles instead of mean
% obj = @(q) ACCtoBoundFit_quantiles_goLocked(exp(q), Target, ...
%                                            dt_ms, StdEv, ...
%                                            p_steps_SP, go_step_SP, TimOutPost, SimNum_fit);

q_hat = fminsearch(obj, q0, opts);

ParamHat = exp(q_hat);

Khat     = ParamHat(1);
BoundHat = ParamHat(2);
NonDechat= ParamHat(3);

fprintf('Fitted single-pulse params:\n');
fprintf('  K     = %.6f\n', Khat);
fprintf('  Bound = %.6f\n', BoundHat);
fprintf('  NonDec= %.6f s\n', NonDechat);

% Khat = 0.45;
% BoundHat = 80;

%% GET MODEL PREDICTIONS FOR SINGLE-PULSE (using fitted params)
[ModelPerf_SP, ModelRT_SP] = simulate_single_pulse_means_goLocked( ...
    Khat, BoundHat, NonDechat, dt_ms, StdEv, StimSet, ...
    p_steps_SP, go_step_SP, TimOutPost, SimNum_fit);

%% SIMULATE DOUBLE-PULSE USING SAME PARAMS
% Assumption consistent with "wait until end": we simulate the same bound process
% but the *observable RT* is fixed to the end of pulse 2 (+ NonDec).
[ModelPerf_DP, ModelRT_DP] = simulate_double_pulse_means_goLocked( ...
    Khat, BoundHat, NonDechat, dt_ms, StdEv, StimSet, ...
    p_steps_DP, gap_steps_DP, go_step_DP, TimOutPost, SimNum_dp);
%% PLOTS
if doPlots
    figure(1); clf;
    plot(log(StimSet), DataPerf, 'ok', 'markerfacecolor','k'); hold on;
    plot(log(StimSet), ModelPerf_SP, 'LineWidth', 2);
    plot(log(StimSet), ModelPerf_DP, '--', 'LineWidth', 2);
    set(gca,'XTick',log(StimSet),'XTickLabel',string(StimSet));
    set(gca,'box','off','tickdir','out');
    xlabel('log(StimStrength)'); ylabel('Performance (P(correct / upper))');
    legend({'Data (single pulse)','Model fit (single pulse)','Model prediction (double pulse)'}, ...
           'Location','southeast');

    figure(2); clf;
    plot(log(StimSet), DataRT, 'ok', 'markerfacecolor','k'); hold on;
    plot(log(StimSet), ModelRT_SP, 'LineWidth', 2);
    plot(log(StimSet), ModelRT_DP, '--', 'LineWidth', 2);
    set(gca,'XTick',log(StimSet),'XTickLabel',string(StimSet));
    set(gca,'box','off','tickdir','out');
    xlabel('log(StimStrength)'); ylabel('Mean RT from GO (s)');
    legend({'Data (single pulse)','Model fit (single pulse)','Model prediction (double pulse)'}, ...
           'Location','northeast');
end

%% END MAIN SCRIPT 

% output:
% Exiting: Maximum number of function evaluations has been exceeded
%          - increase MaxFunEvals option.
%          Current function value: 1.821345 
% 
% Exiting: Maximum number of function evaluations has been exceeded
%          - increase MaxFunEvals option.
%          Current function value: 1.271789 
% 
% Fitted single-pulse params:
%   K     = 1.000494
%   Bound = 67.603942
%   NonDec= 0.598973 s