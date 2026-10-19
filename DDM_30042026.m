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
go_step = round((singlePulseDur_ms + goDelay_ms) / dt_ms);

% Double pulse
p_steps_DP   = round(pulseDur_ms / dt_ms);
gap_steps_DP = round(gapDur_ms / dt_ms);
go_step_DP   = round((pulseDur_ms + gapDur_ms + pulseDur_ms + goDelay_ms) / dt_ms);

% Total simulation Time
T_total = go_step + TimOutPost;

%% NO-SHAPE LIKELIHOOD METHOD - PARAMETERS

cfg.StimSet = [0.016 0.032 0.064 0.128 0.256 0.512];
cfg.tol = 1e-12;

cfg.time_resolution = 5; % ms
cfg.bin_size = cfg.time_resolution/1000; % seconds

cfg.SimNum = 10000;

cfg.singlePulseDur_ms = 120;
cfg.goDelay_ms = 50;

cfg.dt_ms = 0.1;
cfg.StdEv = 1;
cfg.TimOutPost = 4000;

cfg.go_step = round((cfg.singlePulseDur_ms + cfg.goDelay_ms) / cfg.dt_ms);
cfg.p_steps = round(cfg.singlePulseDur_ms / cfg.dt_ms);
cfg.T_total = cfg.go_step + cfg.TimOutPost;


%% FIT WITH NoShape-Likelihood 

Param0 = [0.35, 60, 0.43];

objfun = @(Param) debug_wrapper(Param, ExpData, cfg);

options = optimset('Display','iter', ...
                   'MaxFunEvals',10, ...
                   'MaxIter',10);

[ParamHat, negloglikHat] = fminsearch(objfun, Param0, options);

K_hat_ll    = ParamHat(1);
Bound_hat_ll  = ParamHat(2);
NonDec_hat_ll = ParamHat(3);

fprintf('\nNoShape-Likelihood fit parameters:\n');
fprintf('  K     = %.6f\n', K_hat_ll);
fprintf('  Bound = %.6f\n', Bound_hat_ll);
fprintf('  NonDec= %.6f s\n', NonDec_hat_ll);


%% Simulate trials with no-shape likelihood parameters
[SimResults_hat, RT_hat, Acc_hat, RT_binned_hat] = noshape_simulate(ParamHat, cfg);

% SimResults_hat(j,i,1)   % RT
% SimResults_hat(j,i,2)   % accuracy
% SimResults_hat(j,i,3)   % binned RT

%% %% PLOT SIMULATED VS ACTUAL PERFORMANCE AND RT

%% ACTUAL DATA BINNED RESULTS 

% STORAGE FOR ACTUAL DATA RESULTS

max_trials = max(arrayfun(@(s) sum(abs(ExpData.StimStrength - s) < tol), StimSet));

DataResults = nan(length(StimSet), max_trials, 3);
bin_size = cfg.bin_size;

for j = 1:length(StimSet)

     
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% BINNING ACTUAL RT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        idx_data = abs(ExpData.StimStrength - StimSet(j)) < tol;

        RT_data  = ExpData.RT(idx_data); % here s
        Acc_data = (ExpData.StimDir(idx_data) == ExpData.RespDir(idx_data));
    
        % bin RT (same rule as simulation)
        RT_binned = floor(RT_data / bin_size) * bin_size;
    
        n_trials = length(RT_data);
    
        % store into 3D matrix
        DataResults(j,1:n_trials,1) = RT_data(:)';
        DataResults(j,1:n_trials,2) = Acc_data(:)';
        DataResults(j,1:n_trials,3) = RT_binned(:)';

end

%% PLOT SIMULATED VS ACTUAL
StimSet = cfg.StimSet;

nCoh = length(StimSet);

perf_data = nan(nCoh,1);
perf_sim  = nan(nCoh,1);

rt_data = nan(nCoh,1);
rt_sim  = nan(nCoh,1);

for j = 1:nCoh

    % actual data
    data_acc = squeeze(DataResults(j,:,2));
    data_rt  = squeeze(DataResults(j,:,1));

    valid_data = ~isnan(data_acc) & ~isnan(data_rt);

    perf_data(j) = mean(data_acc(valid_data));
    rt_data(j)   = mean(data_rt(valid_data));

    % simulated data
    sim_acc = squeeze(SimResults_hat(j,:,2));
    sim_rt  = squeeze(SimResults_hat(j,:,1));

    valid_sim = ~isnan(sim_acc) & ~isnan(sim_rt);

    perf_sim(j) = mean(sim_acc(valid_sim));
    rt_sim(j)   = mean(sim_rt(valid_sim));

end

%% PERFORMANCE PLOT

figure
plot(log(StimSet), perf_data, 'o-', 'LineWidth', 2); hold on;
plot(log(StimSet), perf_sim, 's-', 'LineWidth', 2);

set(gca,'XTick',log(StimSet),'XTickLabel',string(StimSet))
set(gca,'box','off','tickdir','out')

xlabel('log(StimStrength)')
ylabel('Proportion correct');
title('Actual vs Simulated Performance');
legend({'Actual data','Simulated model'}, 'Location','best');
box off
grid on;

%% RT PLOT

figure
plot(log(StimSet), rt_data, 'o-', 'LineWidth', 2); hold on;
plot(log(StimSet), rt_sim, 's-', 'LineWidth', 2);

set(gca,'XTick',log(StimSet),'XTickLabel',string(StimSet))
set(gca,'box','off','tickdir','out')

xlabel('log(StimStrength)')
ylabel('Mean RT (s)');
title('Actual vs Simulated Reaction Time');
legend({'Actual data','Simulated model'}, 'Location','best');
box off
grid on;


%% RT DISTRIBUTION SPLIT BY CORRECT / ERROR

figure

for j = 1:length(cfg.StimSet)

    subplot(2,3,j);

    % SIM
    sim_rt  = squeeze(SimResults_hat(j,:,3));
    sim_acc = squeeze(SimResults_hat(j,:,2));

    % force column vectors
    sim_rt  = sim_rt(:);
    sim_acc = sim_acc(:);

    % remove NaNs
    valid_sim = ~isnan(sim_rt) & ~isnan(sim_acc);

    sim_rt  = sim_rt(valid_sim);
    sim_acc = sim_acc(valid_sim);

    % DATA
    data_rt  = squeeze(DataResults(j,:,3));
    data_acc = squeeze(DataResults(j,:,2));

    % force column vectors
    data_rt  = data_rt(:);
    data_acc = data_acc(:);

    % remove NaNs
    valid_data = ~isnan(data_rt) & ~isnan(data_acc);

    data_rt  = data_rt(valid_data);
    data_acc = data_acc(valid_data);

    % SPLIT BY CORRECT / ERROR
    sim_rt_corr = sim_rt(sim_acc == 1);
    sim_rt_err  = sim_rt(sim_acc == 0);

    data_rt_corr = data_rt(data_acc == 1);
    data_rt_err  = data_rt(data_acc == 0);

    % sim_rt_corr   % simulated correct RTs
    % sim_rt_err    % simulated error RTs
    % data_rt_corr  % actual correct RTs
    % data_rt_err   % actual error RTs

    % COMMON BINS
    all_rt = [sim_rt(:); data_rt(:)];

    edges = min(all_rt):cfg.bin_size:max(all_rt);

    bin_centers = edges(1:end-1) + diff(edges)/2;

    % HISTOGRAM COUNTS
    % Count how many RTs fall into a bin

    sim_corr_counts = histcounts(sim_rt_corr, edges, ...
        'Normalization','probability');

    sim_err_counts = histcounts(sim_rt_err, edges, ...
        'Normalization','probability');

    data_corr_counts = histcounts(data_rt_corr, edges, ...
        'Normalization','probability');

    data_err_counts = histcounts(data_rt_err, edges, ...
        'Normalization','probability');

    % PLOT
    plot(bin_centers, data_corr_counts, 'LineWidth',2); 
    hold on;

    plot(bin_centers, data_err_counts, 'LineWidth',2);

    plot(bin_centers, sim_corr_counts, '--', 'LineWidth',2);

    plot(bin_centers, sim_err_counts, '--', 'LineWidth',2);

    title(sprintf('Coherence %.3f', cfg.StimSet(j)));
    xlabel('RT (s)');
    ylabel('Probability');

    legend({'Data Correct','Data Error', ...
            'Model Correct','Model Error'}, ...
            'Location','best');

    grid on;

end