
% This script 
% 1. gets single pulse participant data (trial level) and creates a stim-level data
% 2. fits parameters to single pulse data by using both mse and mle
% 3. simulate single pulse and double pulse trials with the (both set of)
% fitted parameters and plots the results


% functions this script uses:
   
    % to fit parameters to single pulse data with mse:
    % ACCtoBoundFit2_fixRT_goLocked.m (previous one)

    % to fit parameters to single pulse data with mle:
    % ACCtoBoundFit_MLE_manual_jointlikelihood.m

    % to simulate single pulse trials based on the fitted parameters:
    % simulate_single_pulse_means_goLocked_signed.m

    % to simulate double pulse trials with the parameters extracted from
    % the single pulse data: simulate_double_pulse_means_goLocked_signed.m

    
close all; clear;

%% DATA LOAD
load('Data/ExpData.mat');

%% Direction coding to -1/+1 if needed
% Convert stimulus direction (0 / 180 → ±1)
StimDir_signed = zeros(size(ExpData.StimDir));

StimDir_signed(ExpData.StimDir == 0)   =  1;   % 0° motion
StimDir_signed(ExpData.StimDir == 180) = -1;   % 180° motion

% Convert response direction (1 / 2 → ±1)
RespDir_signed = zeros(size(ExpData.RespDir));

RespDir_signed(ExpData.RespDir == 1) =  1;   % button 1
RespDir_signed(ExpData.RespDir == 2) = -1;   % button 2

% Replace in ExpData
ExpData.StimDir = StimDir_signed;
ExpData.RespDir = RespDir_signed;
% ExpData.RespDir = map_to_pm1(ExpData.RespDir);
% ExpData.StimDir = map_to_pm1(ExpData.StimDir);

%% USER SETTINGS
StimSet = [0.016 0.032 0.064 0.128 0.256 0.512];
tol = 1e-12;

% Timing (ms)
singlePulseDur_ms = 120;   % set to your actual single-pulse duration
pulseDur_ms       = 120;   % double-pulse duration for each pulse
gapDur_ms         = 360;   % set to desired double-pulse gap
goDelay_ms        = 50;   % delay from end of last pulse to GO

% Simulation settings
dt_ms = 0.1;   % ms per step
% dt_ms = 0.5;   % for quicker setting
StdEv      = 1;
TimOutPost = 4000;  % post-GO simulation steps

% SimNum_fit = 4000;  % for fitting, you need to use more than before for KDE stability
% SimNum_SP  = 4000;  % for single-pulse predictions
% SimNum_DP  = 4000;  % for double-pulse predictions

% for quicker fitting/testing
SimNum_fit = 200;
SimNum_SP  = 4000;
SimNum_DP  = 4000;

% Initial guess for [K Bound NonDec]
Param0 = [0.4, 32, 0.45];

doPlots = true;

%% BUILD SINGLE-PULSE DATA SUMMARY
DataPerf = nan(numel(StimSet),1);
DataRT   = nan(numel(StimSet),1);

for j = 1:numel(StimSet)
    idx = abs(ExpData.StimStrength - StimSet(j)) < tol;

    Acc_j = (ExpData.StimDir(idx) == ExpData.RespDir(idx));

    DataPerf(j) = mean(Acc_j);
    DataRT(j)   = mean(ExpData.RT(idx));
end

DataSingle = [DataPerf DataRT];

assert(all(~isnan(DataPerf)) && all(~isnan(DataRT)), ...
    'Some StimStrength levels are missing from the data.');

%% STEP COUNTS
% Single pulse
p_steps_SP = round(singlePulseDur_ms / dt_ms);
go_step_SP = round((singlePulseDur_ms + goDelay_ms) / dt_ms);

% Double pulse
p_steps_DP   = round(pulseDur_ms / dt_ms);
gap_steps_DP = round(gapDur_ms / dt_ms);
go_step_DP   = round((pulseDur_ms + gapDur_ms + pulseDur_ms + goDelay_ms) / dt_ms);

% %% FIT WITH MSE
opts = optimset('MaxFunEvals',800,'MaxIter',800,'Display','iter');
% % optimally eval: 3000, but for quicker fitting I have changed
q0 = log(max(Param0,1e-6));

obj_mse = @(q) ACCtoBoundFit2_goLocked_signed(exp(q), DataSingle, ...
    dt_ms, StdEv, StimSet, p_steps_SP, go_step_SP, TimOutPost, SimNum_fit);

qhat_mse = fminsearch(obj_mse, q0, opts);
ParamHat_MSE = exp(qhat_mse);

Khat_MSE      = ParamHat_MSE(1);
BoundHat_MSE  = ParamHat_MSE(2);
NonDechat_MSE = ParamHat_MSE(3);

fprintf('\nMSE fit parameters:\n');
fprintf('  K     = %.6f\n', Khat_MSE);
fprintf('  Bound = %.6f\n', BoundHat_MSE);
fprintf('  NonDec= %.6f s\n', NonDechat_MSE);

%% FIT WITH MLE
% obj_mle = @(q) ACCtoBoundFit_MLE_KDE(exp(q), ExpData, dt_ms, StdEv, StimSet, p_steps_SP, go_step_SP, TimOutPost, SimNum_fit);

obj_mle = @(q) ACCtoBoundFit_MLE_manual_jointlikelihood(exp(q), ExpData, ...
    dt_ms, StdEv, StimSet, p_steps_SP, go_step_SP, TimOutPost, SimNum_fit);

qhat_mle = fminsearch(obj_mle, q0, opts);
ParamHat_MLE = exp(qhat_mle);

Khat_MLE      = ParamHat_MLE(1);
BoundHat_MLE  = ParamHat_MLE(2);
NonDechat_MLE = ParamHat_MLE(3);

fprintf('\nMLE fit parameters:\n');
fprintf('  K     = %.6f\n', Khat_MLE);
fprintf('  Bound = %.6f\n', BoundHat_MLE);
fprintf('  NonDec= %.6f s\n', NonDechat_MLE);


%% SINGLE-PULSE PREDICTIONS FROM BOTH FITS


[ModelPerf_MSE, ModelRT_MSE] = simulate_single_pulse_means_goLocked_signed( ...
    Khat_MSE, BoundHat_MSE, NonDechat_MSE, ...
    dt_ms, StdEv, StimSet, p_steps_SP, go_step_SP, TimOutPost, SimNum_SP);

[ModelPerf_MLE, ModelRT_MLE] = simulate_single_pulse_means_goLocked_signed( ...
    Khat_MLE, BoundHat_MLE, NonDechat_MLE, ...
    dt_ms, StdEv, StimSet, p_steps_SP, go_step_SP, TimOutPost, SimNum_SP);


%% PLOT SINGLE PULSE FITS
figure(1); clf;

subplot(1,2,1)
plot(log(StimSet), DataPerf, 'ok', 'markerfacecolor','k', 'markersize',7); hold on;
%plot(log(StimSet), ModelPerf_MSE, '-o', 'LineWidth',2);
plot(log(StimSet), ModelPerf_MLE, '--o', 'LineWidth',2);
set(gca,'XTick',log(StimSet),'XTickLabel',string(StimSet));
set(gca,'box','off','tickdir','out');
xlabel('log(StimStrength)');
ylabel('Accuracy');
%legend({'Actual single-pulse data','MLE fit'}, 'Location','southeast');
legend({'Actual single-pulse data','MSE fit','MLE fit'}, 'Location','southeast');
title('Single-pulse performance');

subplot(1,2,2)
plot(log(StimSet), DataRT, 'ok', 'markerfacecolor','k', 'markersize',7); hold on;
plot(log(StimSet), ModelRT_MSE, '-o', 'LineWidth',2);
plot(log(StimSet), ModelRT_MLE, '--o', 'LineWidth',2);
set(gca,'XTick',log(StimSet),'XTickLabel',string(StimSet));
set(gca,'box','off','tickdir','out');
xlabel('log(StimStrength)');
ylabel('Mean RT from GO (s)');
%legend({'Actual single-pulse data','MLE fit'}, 'Location','southeast');
legend({'Actual single-pulse data','MSE fit','MLE fit'}, 'Location','northeast');
title('Single-pulse RT');

sgtitle('Single-pulse fit comparison: MSE vs MLE');
%% DOUBLE-PULSE MODEL PREDICTION 
[ModelPerf_DP_MSE, ModelRT_DP_MSE] = simulate_double_pulse_means_goLocked_signed( ...
    Khat_MSE, BoundHat_MSE, NonDechat_MSE, ...
    dt_ms, StdEv, StimSet, p_steps_DP, gap_steps_DP, go_step_DP, TimOutPost, SimNum_DP);

[ModelPerf_DP_MLE, ModelRT_DP_MLE] = simulate_double_pulse_means_goLocked_signed( ...
    Khat_MLE, BoundHat_MLE, NonDechat_MLE, ...
    dt_ms, StdEv, StimSet, p_steps_DP, gap_steps_DP, go_step_DP, TimOutPost, SimNum_DP);
%% PLOT SINGLE- AND DOUBLE-PULSE RESULTS TOGETHER
figure(2); clf;

subplot(1,2,1)
plot(log(StimSet), DataPerf, 'ok', 'markerfacecolor','k', 'markersize',7); hold on;

plot(log(StimSet), ModelPerf_MSE, '-o', 'LineWidth',2, 'MarkerFaceColor','auto');
plot(log(StimSet), ModelPerf_MLE, '--o', 'LineWidth',2, 'MarkerFaceColor','auto');

plot(log(StimSet), ModelPerf_DP_MSE, '-s', 'LineWidth',2, 'MarkerFaceColor','auto');
plot(log(StimSet), ModelPerf_DP_MLE, '--s', 'LineWidth',2, 'MarkerFaceColor','auto');

set(gca,'XTick',log(StimSet),'XTickLabel',string(StimSet));
set(gca,'box','off','tickdir','out');
xlabel('log(StimStrength)');
ylabel('Accuracy');
% legend({'Single-pulse data', ...
%         'Single-pulse fit (MLE)', ...
%         'Double-pulse prediction (MLE params)'}, ...
%         'Location','southeast');
legend({'Single-pulse data', ...
        'Single-pulse fit (MSE)', ...
        'Single-pulse fit (MLE)', ...
        'Double-pulse prediction (MSE params)', ...
        'Double-pulse prediction (MLE params)'}, ...
        'Location','southeast');
title('Performance');

subplot(1,2,2)
plot(log(StimSet), DataRT, 'ok', 'markerfacecolor','k', 'markersize',7); hold on;

plot(log(StimSet), ModelRT_MSE, '-o', 'LineWidth',2, 'MarkerFaceColor','auto');
plot(log(StimSet), ModelRT_MLE, '--o', 'LineWidth',2, 'MarkerFaceColor','auto');

plot(log(StimSet), ModelRT_DP_MSE, '-s', 'LineWidth',2, 'MarkerFaceColor','auto');
plot(log(StimSet), ModelRT_DP_MLE, '--s', 'LineWidth',2, 'MarkerFaceColor','auto');

set(gca,'XTick',log(StimSet),'XTickLabel',string(StimSet));
set(gca,'box','off','tickdir','out');
xlabel('log(StimStrength)');
ylabel('Mean RT from GO (s)');
% legend({'Single-pulse data', ...
%         'Single-pulse fit (MLE)', ...
%         'Double-pulse prediction (MLE params)'}, ...
%         'Location','southeast');
legend({'Single-pulse data', ...
        'Single-pulse fit (MSE)', ...
        'Single-pulse fit (MLE)', ...
        'Double-pulse prediction (MSE params)', ...
        'Double-pulse prediction (MLE params)'}, ...
        'Location','northeast');
title('RT');

sgtitle('Single-pulse fits and double-pulse predictions from MSE vs MLE parameters');

%% PLOT MLE SINGLE + DOUBLE PULSE

figure(3); clf;

subplot(1,2,1)

plot(log(StimSet), DataPerf, 'ok', ...
    'markerfacecolor','k','markersize',7); hold on

plot(log(StimSet), ModelPerf_MLE, '-o', ...
    'LineWidth',2,'MarkerFaceColor','auto')

plot(log(StimSet), ModelPerf_DP_MLE, '--s', ...
    'LineWidth',2,'MarkerFaceColor','auto')

set(gca,'XTick',log(StimSet),'XTickLabel',string(StimSet))
set(gca,'box','off','tickdir','out')

xlabel('log(StimStrength)')
ylabel('Accuracy')

legend({'Single-pulse data', ...
        'Single-pulse model (MLE)', ...
        'Double-pulse prediction (MLE)'}, ...
        'Location','southeast')

title('Performance')


subplot(1,2,2)

plot(log(StimSet), DataRT, 'ok', ...
    'markerfacecolor','k','markersize',7); hold on

plot(log(StimSet), ModelRT_MLE, '-o', ...
    'LineWidth',2,'MarkerFaceColor','auto')

plot(log(StimSet), ModelRT_DP_MLE, '--s', ...
    'LineWidth',2,'MarkerFaceColor','auto')

set(gca,'XTick',log(StimSet),'XTickLabel',string(StimSet))
set(gca,'box','off','tickdir','out')

xlabel('log(StimStrength)')
ylabel('Mean RT from GO (s)')

legend({'Single-pulse data', ...
        'Single-pulse model (MLE)', ...
        'Double-pulse prediction (MLE)'}, ...
        'Location','northeast')

title('Reaction Time')

sgtitle('Single-pulse fit and double-pulse prediction (MLE parameters)')


%% 