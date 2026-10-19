function [NegLogLike] = ACCtoBoundFit_MLE_manual_jointlikelihood(Param, ExpData, dt_ms, StdEv, StimSet, ...
                                                 p_steps, go_step, TimOutPost, SimNum)
% PURPOSE
% 
% This function fits a single-pulse accumulation-to-bound / drift-diffusion model
% using a MANUAL maximum likelihood calculation.
%
% The function returns a single scalar:
%
%   NegLogLike = negative log-likelihood of the observed data under the model
%
% so that MATLAB optimizers such as fminsearch can MINIMIZE this value.
%
%
% MODEL IDEA
% 
% For each stimulus strength:
%   1. Simulate many trials from the model
%   2. From those simulated trials estimate:
%        a) probability of a correct response
%        b) RT density for correct trials
%        c) RT density for error trials
%   3. Ask: how probable are the participant's observed correct/error outcomes
%      together with their RTs under those model-generated probabilities/densities?
%
%
% JOINT LIKELIHOOD USED HERE
% 
% Instead of treating performance and RT as two separate blocks, this version
% makes the likelihood more explicit:
%
%   For correct trials:
%       p(correct, RT | theta) = p(correct | theta) * p(RT | correct, theta)
%
%   For error trials:
%       p(error, RT | theta) = p(error | theta) * p(RT | error, theta)
%
% So the negative log-likelihood is accumulated directly from:
%
%   - log p_corr + log f_corr(RT)   for correct trials
%   - log p_err  + log f_err(RT)    for error trials
%
%
% INPUTS
% 
% Param      : [K Bound NonDec]
%              K       = drift/sensitivity scaling parameter
%              Bound   = symmetric decision threshold
%              NonDec  = non-decision time added to GO-locked RT
%
% ExpData    : struct containing trial-level participant data with fields:
%              ExpData.RT
%              ExpData.StimStrength
%              ExpData.StimDir
%              ExpData.RespDir
%
% dt_ms      : simulation time step in milliseconds
% StdEv      : standard deviation of momentary evidence noise
% StimSet    : vector of stimulus strengths/coherences to fit
% p_steps    : number of time steps during which the stimulus pulse is present
% go_step    : time step at which the GO cue appears
% TimOutPost : number of time steps simulated after GO
% SimNum     : number of simulated trials per stimulus strength
%
%
% OUTPUT
% 
% NegLogLike : scalar negative log-likelihood
%
%
% IMPORTANT TASK ASSUMPTION
% 
% RT is GO-locked:
%   participants are not allowed to respond before the GO signal.
%
% Therefore the modeled RT is:
%
%   RT = max(0, decision_time - go_time) + NonDec
%
% So if the internal decision is finished before GO, the measured RT is just NonDec.


    
    % 1) UNPACK MODEL PARAMETERS
   
    K      = Param(1);
    Bound  = Param(2);
    NonDec = Param(3);

   
    % 2) NUMERICAL SAFETY CONSTANTS
    
    tol         = 1e-12;   % for matching floating-point stimulus strengths
    eps_prob    = 1e-6;    % floor for probabilities such as p(correct)
    eps_rt      = 1e-8;    % floor for RT before log()
    min_log_sd  = 0.05;    % realistic lower bound for lognormal SD

    % Initialize total negative log-likelihood accumulator
    NegLogLike = 0;

    
    % 3) TOTAL SIMULATION LENGTH
    
    T_total = go_step + TimOutPost;

    
    % 4) LOOP OVER EACH STIMULUS STRENGTH / COHERENCE
    
    for j = 1:length(StimSet)

        
        % A. EXTRACT OBSERVED DATA FOR THIS COHERENCE
        
        idx_data = abs(ExpData.StimStrength - StimSet(j)) < tol;

        RT_data      = ExpData.RT(idx_data);
        StimDir_data = ExpData.StimDir(idx_data);
        RespDir_data = ExpData.RespDir(idx_data);

        % Observed trial correctness: 1 = correct, 0 = incorrect
        Acc_data = (StimDir_data == RespDir_data);

        if isempty(RT_data)
            continue;
        end

        
        % B. SIMULATE MODEL TRIALS FOR THIS COHERENCE
        

        
        % B1) BALANCED SIMULATED STIMULUS DIRECTIONS
        
        StimDir_sim = ones(SimNum,1);
        StimDir_sim((SimNum/2+1):end) = -1;

        
        % B2) DEFINE DRIFT FOR EACH SIMULATED TRIAL
        
        mu = K * StimSet(j) * StimDir_sim';   % 1 x SimNum

        
        % B3) GENERATE MOMENTARY EVIDENCE SAMPLES
        
        e = StdEv * sqrt(dt_ms) * randn(T_total, SimNum);
        e(1:p_steps,:) = e(1:p_steps,:) + repmat(mu, p_steps, 1);

        
        % B4) ACCUMULATE EVIDENCE OVER TIME
        
        ACC = cumsum(e);

        
        % B5) FIND UPPER-BOUND HITTING TIME
        
        ACC_pos = ACC;
        ACC_pos(ACC >= Bound) = nan;
        mom_ev_pos = (ACC_pos + e) - ACC_pos;
        ACC_pos = cumsum(mom_ev_pos);
        RT_pos = sum(~isnan(ACC_pos),1) + 1;

        
        % B6) FIND LOWER-BOUND HITTING TIME
        
        ACC_neg = ACC;
        ACC_neg(ACC <= -Bound) = nan;
        mom_ev_neg = (ACC_neg + e) - ACC_neg;
        ACC_neg = cumsum(mom_ev_neg);
        RT_neg = sum(~isnan(ACC_neg),1) + 1;

        
        % B7) FIRST BOUND HIT = EARLIER OF UPPER/LOWER HIT
        
        RT_hit = RT_pos;
        mask = RT_hit > RT_neg;
        RT_hit(mask) = RT_neg(mask);

        
        % B8) HANDLE TRIALS THAT NEVER HIT A BOUND
       
        RT_hit(RT_hit > T_total | isnan(RT_hit)) = T_total;

        
        % B9) DETERMINE MODEL RESPONSE DIRECTION
        
        idx_sim = sub2ind(size(ACC), RT_hit(:), (1:SimNum)');
        RespDir_sim = ones(SimNum,1);
        RespDir_sim(ACC(idx_sim) < 0) = -1;

        
        % B10) DETERMINE MODEL CORRECTNESS
        
        Acc_sim = (RespDir_sim == StimDir_sim);

       
        % B11) COMPUTE GO-LOCKED RT
        
        rt_from_go_steps = RT_hit - go_step;
        rt_from_go_steps(rt_from_go_steps < 0) = 0;
        RT_sim = rt_from_go_steps * dt_ms/1000 + NonDec;

        
        % C. ESTIMATE MODEL PROBABILITIES OF CORRECT / ERROR
        
        p_corr = mean(Acc_sim);
        p_corr = max(min(p_corr, 1 - eps_prob), eps_prob);

        p_err = 1 - p_corr;
        p_err = max(min(p_err, 1 - eps_prob), eps_prob);

        
        % D. SPLIT SIMULATED AND OBSERVED RTs BY OUTCOME CLASS
        
        RT_sim_corr = RT_sim(Acc_sim == 1);
        RT_sim_err  = RT_sim(Acc_sim == 0);

        RT_data_corr = RT_data(Acc_data == 1);
        RT_data_err  = RT_data(Acc_data == 0);

        
        % E. CORRECT TRIALS: joint likelihood
        %    p(correct, RT | theta) = p_corr * p(RT | correct, theta)
        
        if numel(RT_data_corr) > 0 && numel(RT_sim_corr) >= 2

            % Estimate lognormal parameters from simulated correct RTs
            logRT_sim_corr = log(max(RT_sim_corr, eps_rt));
            mu_corr = mean(logRT_sim_corr);
            sd_corr = std(logRT_sim_corr);

            % Prevent unrealistically tiny spread, which causes exploding NLL
            if sd_corr < min_log_sd
                sd_corr = min_log_sd;
            end

            % Protect observed RTs before taking log
            RT_obs_corr = max(RT_data_corr, eps_rt);

            % Conditional log-density:
            % log p(RT | correct, theta)
            logpdf_corr = -log(RT_obs_corr .* sd_corr * sqrt(2*pi)) ...
                          - ((log(RT_obs_corr) - mu_corr).^2) ./ (2 * sd_corr^2);

            % Joint log-likelihood for each correct trial:
            % log p(correct, RT) = log p_corr + log p(RT | correct)
            NegLogLike = NegLogLike - sum(log(p_corr) + logpdf_corr);
        end

        
        % F. ERROR TRIALS: joint likelihood
        %    p(error, RT | theta) = p_err * p(RT | error, theta)
       
        if numel(RT_data_err) > 0 && numel(RT_sim_err) >= 2

            % Estimate lognormal parameters from simulated error RTs
            logRT_sim_err = log(max(RT_sim_err, eps_rt));
            mu_err = mean(logRT_sim_err);
            sd_err = std(logRT_sim_err);

            % Prevent unrealistically tiny spread, which causes exploding NLL
            if sd_err < min_log_sd
                sd_err = min_log_sd;
            end

            % Protect observed RTs before taking log
            RT_obs_err = max(RT_data_err, eps_rt);

            % Conditional log-density:
            % log p(RT | error, theta)
            logpdf_err = -log(RT_obs_err .* sd_err * sqrt(2*pi)) ...
                         - ((log(RT_obs_err) - mu_err).^2) ./ (2 * sd_err^2);

            % Joint log-likelihood for each error trial:
            % log p(error, RT) = log p_err + log p(RT | error)
            NegLogLike = NegLogLike - sum(log(p_err) + logpdf_err);
        end
    end
end