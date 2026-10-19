function [NegLogLike] = ACCtoBoundFit_MLE_manual(Param, ExpData, dt_ms, StdEv, StimSet, ...
                                                 p_steps, go_step, TimOutPost, SimNum)
% ACCtoBoundFit_MLE_manual
%
% PURPOSE
% -------
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
% ----------
% For each stimulus strength:
%   1. Simulate many trials from the model
%   2. From those simulated trials estimate:
%        a) probability of a correct response
%        b) RT density for correct trials
%        c) RT density for error trials
%   3. Ask: how probable are the participant's observed trial outcomes and RTs
%      under those model-generated probabilities/densities?
%
%
% LIKELIHOOD FACTORIZATION
% ------------------------
% For each observed trial, the joint likelihood is written as:
%
%   p(choice, RT | theta) = p(choice | theta) * p(RT | choice, theta)
%
% where theta = [K, Bound, NonDec]
%
% That means:
%   - choice likelihood is handled with a Bernoulli model
%   - RT likelihood is handled with a manually written Gaussian kernel density estimate
%
%
% INPUTS
% ------
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
% ------
% NegLogLike : scalar negative log-likelihood
%
%
% IMPORTANT TASK ASSUMPTION
% -------------------------
% RT is GO-locked:
%   participants are not allowed to respond before the GO signal.
%
% Therefore the modeled RT is:
%
%   RT = max(0, decision_time - go_time) + NonDec
%
% So if the internal decision is finished before GO, the measured RT is just NonDec.


    % ------------------------------------------------------------
    % 1) UNPACK MODEL PARAMETERS
    % ------------------------------------------------------------
    % These are the candidate parameter values currently being tested
    % by the optimizer.
    K      = Param(1);
    Bound  = Param(2);
    NonDec = Param(3);

    % ------------------------------------------------------------
    % 2) NUMERICAL SAFETY CONSTANTS
    % ------------------------------------------------------------
    % tol      : tolerance for matching floating-point stimulus strengths
    % eps_prob : lower bound for probabilities such as p(correct)
    % eps_pdf  : lower bound for RT density values
    %
    % Why do we need eps_prob and eps_pdf?
    % Because log(0) = -Inf, which would break the optimization.
    % These floors prevent impossible/undefined likelihood values.
    tol      = 1e-12;
    eps_prob = 1e-6;
    eps_pdf  = 1e-8;

    % Initialize total negative log-likelihood accumulator
    NegLogLike = 0;

    % ------------------------------------------------------------
    % 3) TOTAL SIMULATION LENGTH
    % ------------------------------------------------------------
    % We simulate:
    %   - evidence during the pulse
    %   - then no signal but continued noisy accumulation
    %   - through GO
    %   - and some additional time after GO
    %
    % The total number of simulated time steps is:
    T_total = go_step + TimOutPost;

    % ------------------------------------------------------------
    % 4) LOOP OVER EACH STIMULUS STRENGTH / COHERENCE
    % ------------------------------------------------------------
    for j = 1:length(StimSet)

        % ========================================================
        % A. EXTRACT OBSERVED DATA FOR THIS COHERENCE
        % ========================================================
        % Select participant trials with the current stimulus strength
        idx_data = abs(ExpData.StimStrength - StimSet(j)) < tol;

        % Pull out observed RT, stimulus direction, and response direction
        RT_data      = ExpData.RT(idx_data);
        StimDir_data = ExpData.StimDir(idx_data);
        RespDir_data = ExpData.RespDir(idx_data);

        % Observed trial correctness:
        % 1 = correct, 0 = incorrect
        %
        % This assumes StimDir and RespDir are already coded consistently
        % (e.g. -1/+1).
        Acc_data = (StimDir_data == RespDir_data);

        % If there are no observed trials at this coherence, skip
        n_data = numel(RT_data);
        if n_data == 0
            continue;
        end

        % ========================================================
        % B. SIMULATE MODEL TRIALS FOR THIS COHERENCE
        % ========================================================

        % --------------------------------------------------------
        % B1) BALANCED SIMULATED STIMULUS DIRECTIONS
        % --------------------------------------------------------
        % Half the simulated trials are direction +1, half are -1.
        %
        % This lets us simulate both alternatives while keeping the number
        % of simulated trials balanced across directions.
        StimDir_sim = ones(SimNum,1);
        StimDir_sim((SimNum/2+1):end) = -1;

        % --------------------------------------------------------
        % B2) DEFINE DRIFT FOR EACH SIMULATED TRIAL
        % --------------------------------------------------------
        % Drift is proportional to:
        %   K * coherence * stimulus direction
        %
        % So:
        %   positive direction trials get positive drift
        %   negative direction trials get negative drift
        mu = K * StimSet(j) * StimDir_sim';   % 1 x SimNum

        % --------------------------------------------------------
        % B3) GENERATE MOMENTARY EVIDENCE SAMPLES
        % --------------------------------------------------------
        % e is a matrix of size:
        %   [time steps x simulated trials]
        %
        % First we generate Gaussian noise everywhere.
        % Then we add the drift ONLY during the stimulus pulse.
        %
        % This implements:
        %   during pulse:    e_t ~ N(mu, sigma^2)
        %   after pulse:     e_t ~ N(0,  sigma^2)
        e = StdEv * sqrt(dt_ms) * randn(T_total, SimNum);

        % Add drift only during pulse period
        e(1:p_steps,:) = e(1:p_steps,:) + repmat(mu, p_steps, 1);

        % --------------------------------------------------------
        % B4) ACCUMULATE EVIDENCE OVER TIME
        % --------------------------------------------------------
        % The decision variable ACC is the cumulative sum of momentary evidence.
        %
        % This is the discrete-time version of an accumulation-to-bound model:
        %
        %   x_{t+1} = x_t + e_t
        %
        ACC = cumsum(e);

        % --------------------------------------------------------
        % B5) FIND UPPER-BOUND HITTING TIME
        % --------------------------------------------------------
        % We use the same NaN-based logic as in my earlier code:
        % once the upper bound is exceeded, those later samples are set to NaN,
        % then the number of non-NaN entries gives the first hit time.
        ACC_pos = ACC;
        ACC_pos(ACC >= Bound) = nan;

        mom_ev_pos = (ACC_pos + e) - ACC_pos;
        ACC_pos = cumsum(mom_ev_pos);

        RT_pos = sum(~isnan(ACC_pos),1) + 1;

        % --------------------------------------------------------
        % B6) FIND LOWER-BOUND HITTING TIME
        % --------------------------------------------------------
        ACC_neg = ACC;
        ACC_neg(ACC <= -Bound) = nan;

        mom_ev_neg = (ACC_neg + e) - ACC_neg;
        ACC_neg = cumsum(mom_ev_neg);

        RT_neg = sum(~isnan(ACC_neg),1) + 1;

        % --------------------------------------------------------
        % B7) FIRST BOUND HIT = EARLIER OF UPPER/LOWER HIT
        % --------------------------------------------------------
        RT_hit = RT_pos;
        mask = RT_hit > RT_neg;
        RT_hit(mask) = RT_neg(mask);

        % --------------------------------------------------------
        % B8) HANDLE TRIALS THAT NEVER HIT A BOUND
        % --------------------------------------------------------
        % If a trial never reaches either bound within the total simulated
        % time window, its hit time would go out of range.
        %
        % We clamp such trials to the final simulated step.
        RT_hit(RT_hit > T_total | isnan(RT_hit)) = T_total;

        % --------------------------------------------------------
        % B9) DETERMINE MODEL RESPONSE DIRECTION
        % --------------------------------------------------------
        % To know the model's choice on each trial, we read the sign of the
        % decision variable ACC at the decision time RT_hit.
        %
        % If ACC is positive -> choose +1
        % If ACC is negative -> choose -1
        idx_sim = sub2ind(size(ACC), RT_hit(:), (1:SimNum)');

        RespDir_sim = ones(SimNum,1);
        RespDir_sim(ACC(idx_sim) < 0) = -1;

        % --------------------------------------------------------
        % B10) DETERMINE MODEL CORRECTNESS
        % --------------------------------------------------------
        % A simulated trial is correct when the model's response direction
        % matches the stimulus direction.
        Acc_sim = (RespDir_sim == StimDir_sim);

        % --------------------------------------------------------
        % B11) COMPUTE GO-LOCKED RT
        % --------------------------------------------------------
        % Internal decision time may occur before or after GO.
        %
        % If decision happens before GO:
        %   RT from GO = 0
        %
        % If decision happens after GO:
        %   RT from GO = decision_time - go_time
        %
        % Finally we add NonDec.
        rt_from_go_steps = RT_hit - go_step;
        rt_from_go_steps(rt_from_go_steps < 0) = 0;

        RT_sim = rt_from_go_steps * dt_ms/1000 + NonDec;

        % ========================================================
        % C. CHOICE / ACCURACY LIKELIHOOD
        % ========================================================

        % --------------------------------------------------------
        % C1) ESTIMATE MODEL PROBABILITY OF CORRECT RESPONSE
        % --------------------------------------------------------
        % This is the Bernoulli parameter p(correct | theta) for this coherence.
        p_corr = mean(Acc_sim);

        % Apply floor/ceiling to avoid log(0)
        p_corr = max(min(p_corr, 1-eps_prob), eps_prob);

        % --------------------------------------------------------
        % C2) BERNOULLI LOG-LIKELIHOOD FOR OBSERVED CHOICES
        % --------------------------------------------------------
        % Each observed trial contributes:
        %
        %   if correct:   log(p_corr)
        %   if error:     log(1 - p_corr)
        %
        % Mathematically:
        %
        %   log L_choice
        %   = sum_i [ y_i log(p_corr) + (1-y_i) log(1-p_corr) ]
        %
        % where y_i = 1 for correct, 0 for error.
        NegLogLike = NegLogLike ...
            - sum( Acc_data .* log(p_corr) + (1-Acc_data) .* log(1-p_corr) );

        % ========================================================
        % D. RT LIKELIHOOD
        % ========================================================
        % Here we estimate the RT density manually using Gaussian KDE.
        %
        % IMPORTANT:
        % We estimate conditional densities:
        %
        %   f_corr(rt) = p(rt | correct, theta)
        %   f_err(rt)  = p(rt | error,  theta)
        %
        % The class probabilities p(correct) and p(error) were already handled
        % in the Bernoulli part above.
        %
        % So the RT density should be normalized WITHIN EACH CLASS separately,
        % not across correct and error jointly.

        % Simulated RTs for correct and error trials
        RT_sim_corr = RT_sim(Acc_sim == 1);
        RT_sim_err  = RT_sim(Acc_sim == 0);

        % Observed RTs for correct and error trials
        RT_data_corr = RT_data(Acc_data == 1);
        RT_data_err  = RT_data(Acc_data == 0);
        %% MLE with KDE
        % % --------------------------------------------------------
        % % D1) CORRECT RT DENSITY
        % % --------------------------------------------------------
        % % We estimate the density at each observed correct RT using a manual
        % % Gaussian KDE based on the simulated correct RT samples.
        % %
        % % This approximates:
        % %
        % %   f_corr(t)
        % %   = (1 / (n h)) * sum_k phi((t - r_k)/h)
        % %
        % % where:
        % %   r_k = simulated correct RT samples
        % %   h   = bandwidth
        % %   phi = standard normal density
        % if numel(RT_data_corr) > 0 && numel(RT_sim_corr) >= 30
        %     pdf_corr = manual_kde_pdf(RT_sim_corr, RT_data_corr, eps_pdf);
        % 
        %     % Add negative log-likelihood contribution of observed correct RTs
        %     NegLogLike = NegLogLike - sum(log(pdf_corr));
        % end
        % 
        % % --------------------------------------------------------
        % % D2) ERROR RT DENSITY
        % % --------------------------------------------------------
        % if numel(RT_data_err) > 0 && numel(RT_sim_err) >= 30
        %     pdf_err = manual_kde_pdf(RT_sim_err, RT_data_err, eps_pdf);
        % 
        %     % Add negative log-likelihood contribution of observed error RTs
        %     NegLogLike = NegLogLike - sum(log(pdf_err));
        % end

        %% MLE with Lognormal RT Likelihood

        % --------------------------------------------------------
        % D1) CORRECT RT DENSITY: Lognormal approximation
        % --------------------------------------------------------
        if numel(RT_data_corr) > 0 && numel(RT_sim_corr) >= 2 % if we have enough correct RTs and simulated correct RTs (at least 2 needed beacuse we neeed to calculate sd)
            
            % make sure log RT_sim_corr is positive (bigger than epsilon)
            logRT_sim_corr = log(max(RT_sim_corr, eps_pdf)); % we will model log RT as gaussian, and not directly RT
            mu_corr = mean(logRT_sim_corr);
            sd_corr = std(logRT_sim_corr);

            if sd_corr < eps_pdf
                sd_corr = eps_pdf;
            end
            
            RT_obs_corr = max(RT_data_corr, eps_pdf);

            % Lognormal log-density
            logpdf_corr = -log(RT_obs_corr .* sd_corr * sqrt(2*pi)) ...
                          - ((log(RT_obs_corr) - mu_corr).^2) ./ (2 * sd_corr^2);

            NegLogLike = NegLogLike - sum(logpdf_corr);
        end

        % --------------------------------------------------------
        % D2) ERROR RT DENSITY: Lognormal approximation
        % --------------------------------------------------------
        if numel(RT_data_err) > 0 && numel(RT_sim_err) >= 2

            logRT_sim_err = log(max(RT_sim_err, eps_pdf));
            mu_err = mean(logRT_sim_err);
            sd_err = std(logRT_sim_err);

            if sd_err < eps_pdf
                sd_err = eps_pdf;
            end

            RT_obs_err = max(RT_data_err, eps_pdf);

            % Lognormal log-density
            logpdf_err = -log(RT_obs_err .* sd_err * sqrt(2*pi)) ...
                         - ((log(RT_obs_err) - mu_err).^2) ./ (2 * sd_err^2);

            NegLogLike = NegLogLike - sum(logpdf_err);
        end

    end
end