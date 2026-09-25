function [DOSY_X, D_space, info] = TRAIn_DOSY_MFV31(Z, G2, D_params, options)
%TRAIn_DOSY_MF Multi-frequency / multi-component DOSY inversion (V3.1)
%
%   Modelo: Z ≈ K·S·A = Σ_k (K·h_k)·a_k^T
%   Salida DOSY por frecuencia: DOSY_X(:,f) = S·A(:,f)
%
%   V3.1 consolida y mejora:
%     - Estabilidad de pesos sparse adaptativos
%     - Reinicialización robusta de componentes degenerados
%     - Suavizado con reflexión en bordes
%     - Selección automática de r con hold-out (BIC/CV)
%     - Métricas de consistencia enriquecidas (incluye separación de componentes)
%     - Validación de entradas mejorada
%     - Corrección de bug en init_nmf_style para r > size(H)
%
% SINTAXIS:
%   [DOSY_X, D_space, info] = TRAIn_DOSY_MF(Z, G2, D_params)
%   [DOSY_X, D_space, info] = TRAIn_DOSY_MF(Z, G2, D_params, options)
%
% ENTRADAS:
%   Z        : (nGrad x nFreq) matriz de atenuaciones
%   G2       : (nGrad x 1) parámetro experimental (b-value en s/m²)
%   D_params : [Dmin Dmax nD] con D en unidades de (1e-9 m²/s)
%   options  : struct con opciones (ver OPCIONES abajo)
%
% SALIDAS:
%   DOSY_X   : (nD x nFreq) mapa DOSY por frecuencia
%   D_space  : (nD x 1) grid de D en (1e-9 m²/s)
%   info     : struct con campos:
%       .version            - '3.1'
%       .timestamp          - fecha/hora de ejecución
%       .kernel_stats       - diagnósticos del kernel (expo_max, K_min, K_max, decay_*)
%       .S                  - (nD x r) distribuciones de difusión por componente
%       .A                  - (r x nP) contribuciones espectrales
%       .R2_mean_initial    - R² medio inicial
%       .R2_mean_final      - R² medio final
%       .R2_median_final    - R² mediana final
%       .R2_std_final       - desviación estándar de R²
%       .R2_processed_final - R² por frecuencia procesada
%       .consistency        - struct con métricas detalladas
%       .consistency_score  - score global [0,1]
%       .obj_hist           - historial del objetivo
%       .R2_hist            - historial de R²
%       .process_idx        - índices de frecuencias procesadas
%       .n_components       - número de componentes r
%       .n_iterations       - iteraciones realizadas
%       .reinit_used_total  - reinicializaciones totales
%       .time_total         - tiempo de ejecución (s)
%       .similarity_matrix  - matriz de similitud entre frecuencias
%
% OPCIONES PRINCIPALES:
%   mode              : 'polymer' (suave) | 'discrete' (picos estrechos) [default: 'polymer']
%   n_components      : número de componentes r, o 'auto' [default: 3]
%   auto_r            : true para selección automática de r [default: false]
%   r_selection_method: 'bic' | 'cv' [default: 'bic']
%   r_max             : máximo r para auto-selección [default: 8]
%   max_outer_iter    : iteraciones externas máximas [default: 30]
%   outer_tol         : tolerancia de convergencia [default: 1e-4]
%   convergence_window: ventana para criterio de parada [default: 3]
%   max_iter          : iteraciones TRAIn internas [default: 500]
%   lambda_S_smooth   : regularización suavidad (auto por modo)
%   lambda_S_sparse   : regularización sparsity (auto por modo)
%   warm_start        : usar solución previa como inicialización [default: true]
%   init_method       : 'farthest' | 'nmf' | 'kmeans' | 'random' [default: 'farthest']
%   L_order           : 1 (primera diferencia) | 2 (segunda diferencia) [auto por modo]
%   auto_scale_G2     : auto-escalar G2 si kernel degenerado [default: true]
%   verbose           : mostrar progreso [default: true]
%
% EJEMPLO:
%   opts = struct('mode', 'polymer', 'n_components', 'auto', 'verbose', true);
%   [DOSY, D, info] = TRAIn_DOSY_MF(Z, G2, [0.1 100 128], opts);
%
% Autor: Francisco Arrabal-Campos - Universidad de Almería - NMRMBC Research Group
% Versión: 3.1 (Enero 2026)

t0 = tic;

%% ==================== VALIDACIÓN DE ENTRADAS ====================
if nargin < 3
    error('TRAIn_DOSY_MF:InvalidInput', ...
        'Se requieren al menos 3 argumentos: Z, G2, D_params');
end

if nargin < 4 || isempty(options)
    options = struct();
end

Z = double(Z);
G2 = double(G2(:));

[nG, nF] = size(Z);

if numel(G2) ~= nG
    error('TRAIn_DOSY_MF:InvalidInput', ...
        'G2 debe tener longitud nGrad (size(Z,1)). Tiene %d, esperado %d.', numel(G2), nG);
end

if numel(D_params) < 3
    error('TRAIn_DOSY_MF:InvalidInput', ...
        'D_params debe ser [Dmin Dmax nD]. Tiene %d elementos.', numel(D_params));
end

%% ==================== INICIALIZAR INFO ====================
info = struct();
info.version = '3.1';
info.timestamp = datestr(now);
info.kernel_stats = struct();
info.diagnostics = struct();

%% ==================== DEFAULTS ====================
defaults = struct( ...
    'mode', 'polymer', ...
    'n_components', 3, ...
    'auto_r', false, ...
    'r_max', 8, ...
    'r_val_frac', 0.25, ...
    'r_quick_iters', 5, ...
    'r_selection_method', 'bic', ...
    'r_bic_beta', 1.0, ...
    'max_outer_iter', 30, ...
    'outer_tol', 1e-4, ...
    'convergence_window', 3, ...
    'max_iter', 500, ...
    'term_factor', 1.05, ...
    'signal_mode', 'abs', ...
    'normalize_y', true, ...
    'normalize_components', true, ...
    'min_amplitude', 0, ...
    'noise_threshold', 30000, ...
    'rsdt', [], ...
    'init_topM', 256, ...
    'init_method', 'farthest', ...
    'lambda_A', 1e-6, ...
    'spectral_smooth_window', 3, ...
    'lambda_S_smooth', [], ...
    'lambda_S_sparse', [], ...
    'auto_lambda', true, ...
    'auto_lambda_gain', 0.3, ...
    'lambda_bounds', [1e-10, 1e2], ...
    'A_solver', 'ridge_proj', ...
    'G2_scale', 1, ...
    'auto_scale_G2', true, ...
    'verbose', true, ...
    'interpolate_result', false, ...
    'L_order', [], ...
    'rew_epsilon_abs', 1e-12, ...
    'rew_epsilon_rel', 1e-6, ...
    'rew_wmax', 1e6, ...
    'degenerate_threshold', 1e-6, ...
    'min_component_energy', 1e-6, ...
    'reinit_max_per_iter', 2, ...
    'warm_start', true ...
);

options = merge_defaults(options, defaults);

% Alias de compatibilidad
if isfield(options, 'auto_select_r') && ~isempty(options.auto_select_r)
    options.auto_r = logical(options.auto_select_r);
end
if isfield(options, 'auto_select_r_method') && ~isempty(options.auto_select_r_method)
    options.r_selection_method = options.auto_select_r_method;
end

%% ==================== LAMBDAS POR MODO ====================
if isempty(options.lambda_S_smooth)
    if strcmpi(options.mode, 'polymer')
        options.lambda_S_smooth = 1e-2;
    else
        options.lambda_S_smooth = 1e-4;
    end
end

if isempty(options.lambda_S_sparse)
    if strcmpi(options.mode, 'discrete')
        options.lambda_S_sparse = 1e-3;
    else
        options.lambda_S_sparse = 0;
    end
end

if isempty(options.L_order)
    if strcmpi(options.mode, 'polymer')
        options.L_order = 2;
    else
        options.L_order = 1;
    end
end

%% ==================== GRID D Y KERNEL ====================
Dmin = D_params(1);
Dmax = D_params(2);
nD = round(D_params(3));

D_space = logspace(log10(max(Dmin, eps)), log10(max(Dmax, eps)), nD).';
D_SI = D_space * 1e-9;

G2s = G2 * options.G2_scale;
expo_max = max(G2s) * max(D_SI);

info.kernel_stats.original_expo_max = expo_max;
info.kernel_stats.G2_scale_applied = 1.0;

if options.auto_scale_G2 && expo_max < 1e-3
    scale_guess = 10^(ceil(log10(1 / max(expo_max, realmin))));
    scale_guess = min(max(scale_guess, 1e-6), 1e12);
    G2s = G2s * scale_guess;
    info.kernel_stats.G2_scale_applied = scale_guess;
    
    if options.verbose
        fprintf('[TRAIn_DOSY_MF V3.1] auto_scale_G2: x%.3g (expo_max %.3g -> %.3g)\n', ...
            scale_guess, expo_max, max(G2s) * max(D_SI));
    end
end

K = exp(-G2s * D_SI.');
decay_ratio = K(end, :) ./ (K(1, :) + eps);

info.kernel_stats.expo_max = max(G2s) * max(D_SI);
info.kernel_stats.K_min = min(K(:));
info.kernel_stats.K_max = max(K(:));
info.kernel_stats.decay_min = min(decay_ratio);
info.kernel_stats.decay_max = max(decay_ratio);
info.kernel_stats.kernel_ok = info.kernel_stats.decay_min < 0.99;

if options.verbose
    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('   TRAIn-DOSY-MF V3.1 (Multi-Frequency Multi-Component)     \n');
    fprintf('============================================================\n');
    fprintf('\n--- DATOS ---\n');
    fprintf('Z: %d x %d (n_grad x n_freq)\n', nG, nF);
    fprintf('D-space: [%.3g, %.3g] x1e-9 m^2/s (%d pts)\n', Dmin, Dmax, nD);
    fprintf('Modo: %s\n', options.mode);
    fprintf('\n--- KERNEL ---\n');
    fprintf('expo_max: %.4g\n', info.kernel_stats.expo_max);
    fprintf('K range: [%.4g, %.4g]\n', info.kernel_stats.K_min, info.kernel_stats.K_max);
    fprintf('Decay ratio: [%.4f, %.4f]\n', info.kernel_stats.decay_min, info.kernel_stats.decay_max);
    if ~info.kernel_stats.kernel_ok
        fprintf('WARNING: Kernel sin decaimiento (K~1). Verificar unidades.\n');
    else
        fprintf('Kernel OK\n');
    end
end

%% ==================== PREPROCESADO ====================
Zuse = Z;
if strcmpi(options.signal_mode, 'abs')
    Zuse = abs(Zuse);
else
    Zuse = real(Zuse);
end

Zuse = Zuse - min(Zuse, [], 1);
Zuse = max(Zuse, 0);

if options.normalize_y
    denom = max(Zuse(1, :), eps);
    Zuse = Zuse ./ denom;
end

%% ==================== DETECCIÓN DE SEÑAL ====================
if ~isempty(options.rsdt)
    rvec = options.rsdt(:).';
    if numel(rvec) ~= nF
        error('TRAIn_DOSY_MF:InvalidInput', ...
            'options.rsdt debe tener longitud nFreq (%d). Tiene %d.', nF, numel(rvec));
    end
    mask = (rvec >= options.noise_threshold);
else
    amp = max(Zuse, [], 1);
    mask = (amp >= options.min_amplitude);
end

process_idx = find(mask & all(isfinite(Zuse), 1));
Zp = Zuse(:, process_idx);
nP = numel(process_idx);

if options.verbose
    fprintf('\n--- FASE 1: DETECCION ---\n');
    fprintf('Frecuencias con senal: %d / %d (%.1f%%)\n', nP, nF, 100*nP/max(nF,1));
end

if nP < 2
    DOSY_X = zeros(nD, nF);
    info.time_total = toc(t0);
    info.process_idx = process_idx;
    info.R2_mean_initial = 0;
    info.R2_mean_final = 0;
    info.S = [];
    info.A = [];
    warning('TRAIn_DOSY_MF:InsufficientSignal', 'Menos de 2 frecuencias con senal.');
    return;
end

%% ==================== OPERADOR DE REGULARIZACIÓN ====================
if options.L_order == 2
    L = build_second_diff_reflect(nD);
else
    L = build_first_diff(nD);
end

%% ==================== SELECCIÓN DE r ====================
r = options.n_components;
if ischar(r) || isstring(r)
    if strcmpi(string(r), 'auto')
        options.auto_r = true;
    else
        error('TRAIn_DOSY_MF:InvalidInput', ...
            'options.n_components debe ser numerico o ''auto''. Recibido: %s', char(r));
    end
end

if options.auto_r
    r = select_r_auto(Zp, K, L, options);
end

r = max(1, min(round(r), min(options.r_max, nP)));
options.n_components = r;

%% ==================== INICIALIZACIÓN ====================
if options.verbose
    fprintf('\n--- FASE 2: INICIALIZACION ---\n');
    fprintf('Metodo: %s, r = %d\n', char(options.init_method), r);
end

M = min(options.init_topM, nP);
[~, ord] = sort(sum(Zp, 1), 'descend');
sel = ord(1:M);

Hinit = zeros(nD, M);
for ii = 1:M
    y = Zp(:, sel(ii));
    Hinit(:, ii) = lsqnonneg(K, y);
end

switch lower(string(options.init_method))
    case 'nmf'
        S = init_nmf_style(Hinit, r);
    case 'kmeans'
        S = init_centroids_kmeans(Hinit, r);
    case 'random'
        idx_rand = randperm(M, min(r, M));
        S = max(Hinit(:, idx_rand), 0);
        for k = 1:size(S, 2)
            S(:, k) = S(:, k) / (sum(S(:, k)) + eps);
        end
    otherwise
        S = init_centroids_farthest(Hinit, r);
end

S = max(S, 0);

% Asegurar r columnas
if size(S, 2) < r
    n_extra = r - size(S, 2);
    S_extra = rand(nD, n_extra);
    for k = 1:n_extra
        S_extra(:, k) = S_extra(:, k) / (sum(S_extra(:, k)) + eps);
    end
    S = [S, S_extra];
end

if options.normalize_components
    for k = 1:r
        c = sum(S(:, k)) + eps;
        S(:, k) = S(:, k) / c;
    end
end

B = K * S;
A = update_A(B, Zp, options.lambda_A, options.A_solver);
A = max(A, 0);
A = smooth_A_reflect(A, options.spectral_smooth_window);

Yfit = B * A;
R2 = r2_columns(Zp, Yfit);
info.R2_processed_initial = R2;
info.R2_mean_initial = mean(R2);

if options.verbose
    fprintf('R2 inicial: %.4f\n', info.R2_mean_initial);
end

%% ==================== PREPARACIÓN ====================
w_sparse = ones(nD, r);
h_prev = S;

if options.auto_lambda
    dZ = diff(Zp, 1, 1);
    sigma_hat = median(abs(dZ(:))) / 0.6745 / sqrt(2);
    target_res = sigma_hat * sqrt(numel(Zp));
else
    target_res = NaN;
end

%% ==================== OPTIMIZACIÓN ALTERNANTE ====================
if options.verbose
    fprintf('\n--- FASE 3: OPTIMIZACION ---\n');
end

obj_hist = zeros(options.max_outer_iter, 1);
R2_hist = zeros(options.max_outer_iter, 1);
reinit_used_total = 0;
conv_hist = nan(options.convergence_window, 1);

for it = 1:options.max_outer_iter
    
    % Actualizar A
    B = K * S;
    A = update_A(B, Zp, options.lambda_A, options.A_solver);
    A = max(A, 0);
    A = smooth_A_reflect(A, options.spectral_smooth_window);
    
    Yfit = B * A;
    Res = Zp - Yfit;
    
    res_norm_f = sqrt(sum(Res.^2, 1));
    [~, worst_f] = max(res_norm_f);
    
    % Actualizar S componente por componente
    any_reinit = false;
    reinit_this_iter = 0;
    
    for k = 1:r
        ak_old = A(k, :);
        den = sum(ak_old.^2) + eps;
        
        energy_k = (sum(S(:, k)) + eps) * (sum(ak_old) + eps);
        
        is_degen = (energy_k < options.min_component_energy) || ...
                   (max(S(:, k)) < options.degenerate_threshold) || ...
                   (den <= 1e-12);
        
        if is_degen
            if reinit_this_iter < options.reinit_max_per_iter
                yseed = Res(:, worst_f) + B(:, k) * ak_old(worst_f);
                yseed = max(yseed, 0);
                
                h0 = [];
                if options.warm_start
                    h0 = h_prev(:, k);
                end
                
                hk_seed = train_component_traincore(yseed, K, L, ...
                    options.lambda_S_smooth, options.lambda_S_sparse, ones(nD, 1), ...
                    options.max_iter, options.term_factor, h0, options);
                
                hk_seed = max(hk_seed, 0);
                
                if options.normalize_components
                    c = sum(hk_seed) + eps;
                    hk_seed = hk_seed / c;
                    A(k, :) = 0;
                    A(k, worst_f) = max(0.05, 0.1 * max(A(:) + eps)) * c;
                else
                    A(k, :) = 0;
                    A(k, worst_f) = max(0.05, 0.1 * max(A(:) + eps));
                end
                
                Bold = B(:, k);
                S(:, k) = hk_seed;
                B(:, k) = K * S(:, k);
                ak_new = A(k, :);
                
                Yfit = Yfit - Bold * ak_old + B(:, k) * ak_new;
                Res = Zp - Yfit;
                
                res_norm_f = sqrt(sum(Res.^2, 1));
                [~, worst_f] = max(res_norm_f);
                
                any_reinit = true;
                reinit_this_iter = reinit_this_iter + 1;
                reinit_used_total = reinit_used_total + 1;
                
                h_prev(:, k) = hk_seed;
            end
            continue;
        end
        
        Rk = Res + B(:, k) * ak_old;
        yk = max((Rk * ak_old.') / den, 0);
        
        if strcmpi(options.mode, 'discrete') && options.lambda_S_sparse > 0
            w_sparse(:, k) = stable_reweight(S(:, k), options);
        else
            w_sparse(:, k) = ones(nD, 1);
        end
        
        h0 = [];
        if options.warm_start
            h0 = h_prev(:, k);
        end
        
        hk = train_component_traincore(yk, K, L, ...
            options.lambda_S_smooth, options.lambda_S_sparse, w_sparse(:, k), ...
            options.max_iter, options.term_factor, h0, options);
        
        hk = max(hk, 0);
        
        ak_new = ak_old;
        if options.normalize_components
            c = sum(hk) + eps;
            hk = hk / c;
            ak_new = ak_old * c;
            A(k, :) = ak_new;
        end
        
        Bold = B(:, k);
        S(:, k) = hk;
        B(:, k) = K * S(:, k);
        h_prev(:, k) = hk;
        
        Yfit = Yfit - Bold * ak_old + B(:, k) * ak_new;
        Res = Zp - Yfit;
    end
    
    if any_reinit
        B = K * S;
        A = update_A(B, Zp, options.lambda_A, options.A_solver);
        A = max(A, 0);
        A = smooth_A_reflect(A, options.spectral_smooth_window);
        Yfit = B * A;
        Res = Zp - Yfit;
    end
    
    obj = 0.5 * norm(Res, 'fro')^2;
    obj_hist(it) = obj;
    
    R2 = r2_columns(Zp, Yfit);
    R2m = mean(R2);
    R2_hist(it) = R2m;
    
    if options.auto_lambda && isfinite(target_res) && target_res > 0
        res_norm = norm(Res, 'fro');
        factor = (target_res / max(res_norm, eps)) ^ options.auto_lambda_gain;
        options.lambda_S_smooth = clamp(options.lambda_S_smooth * factor, ...
            options.lambda_bounds(1), options.lambda_bounds(2));
    end
    
    if options.verbose
        fprintf('Iter %2d/%d | obj=%.3e | R2=%.4f | lambdaS=%.2e | reinit=%d\n', ...
            it, options.max_outer_iter, obj, R2m, options.lambda_S_smooth, reinit_this_iter);
    end
    
    if options.convergence_window <= 1
        if it > 1
            rel = abs(obj_hist(it-1) - obj) / max(abs(obj_hist(it-1)), eps);
            if rel < options.outer_tol
                if options.verbose
                    fprintf('Convergencia (rel=%.2e < tol)\n', rel);
                end
                break;
            end
        end
    else
        conv_hist = [obj; conv_hist(1:end-1)];
        if it > options.convergence_window && all(isfinite(conv_hist))
            rel_change = abs(conv_hist(1) - conv_hist(end)) / max(abs(conv_hist(end)), eps);
            if rel_change < options.outer_tol
                if options.verbose
                    fprintf('Convergencia (rel_window=%.2e < tol)\n', rel_change);
                end
                break;
            end
        end
    end
end

%% ==================== RESULTADOS FINALES ====================
B = K * S;
A = max(A, 0);
Yfit = B * A;
R2 = r2_columns(Zp, Yfit);

info.R2_processed_final = R2;
info.R2_mean_final = mean(R2);
info.R2_median_final = median(R2);
info.R2_std_final = std(R2);

cons = compute_consistency_metrics(Zp, Yfit, A, S);
info.consistency = cons;
info.consistency_score = cons.score;

DOSY_X = zeros(nD, nF);
DOSY_X(:, process_idx) = S * A;

if options.interpolate_result
    miss = find(~mask);
    if ~isempty(miss) && ~isempty(process_idx)
        x = process_idx(:);
        for j = 1:nD
            v = DOSY_X(j, x);
            DOSY_X(j, miss) = interp1(x, v, miss, 'pchip', 0);
        end
    end
end

[~, simSubIdx] = subsample_indices(nP, 800);
A_sub = A(:, simSubIdx);
Smat = cosine_similarity_cols(A_sub);
info.similarity_matrix = Smat;
info.similarity_matrix_idx = process_idx(simSubIdx);

info.mode = options.mode;
info.n_components = r;
info.process_idx = process_idx;
info.freq_mask = mask;
info.n_processed = nP;
info.S = S;
info.A = A;
info.obj_hist = obj_hist(1:it);
info.R2_hist = R2_hist(1:it);
info.n_iterations = it;
info.lambda_A = options.lambda_A;
info.lambda_S_smooth = options.lambda_S_smooth;
info.lambda_S_sparse = options.lambda_S_sparse;
info.reinit_used_total = reinit_used_total;
info.L_order = options.L_order;
info.init_method = char(options.init_method);
info.time_total = toc(t0);
info.options = options;

if options.verbose
    fprintf('\n============================================================\n');
    fprintf('   RESUMEN FINAL\n');
    fprintf('============================================================\n');
    fprintf('Tiempo total: %.2f s\n', info.time_total);
    fprintf('Iteraciones: %d\n', it);
    fprintf('Componentes: %d\n', r);
    fprintf('R2 inicial: %.4f -> R2 final: %.4f\n', info.R2_mean_initial, info.R2_mean_final);
    fprintf('Consistencia: %.4f\n', info.consistency_score);
    fprintf('Separacion componentes: %.4f\n', cons.component_separation);
    fprintf('Reinicializaciones: %d\n', reinit_used_total);
    fprintf('============================================================\n');
end

end

%% ========================================================================
%%  FUNCIONES AUXILIARES
%% ========================================================================

function out = merge_defaults(in, defaults)
    out = in;
    fn = fieldnames(defaults);
    for i = 1:numel(fn)
        f = fn{i};
        if ~isfield(out, f) || isempty(out.(f))
            out.(f) = defaults.(f);
        end
    end
end

function x = clamp(x, lo, hi)
    x = max(lo, min(hi, x));
end

function L = build_first_diff(n)
    e = ones(n, 1);
    L = spdiags([-e e], [0 1], n-1, n);
end

function L = build_second_diff_reflect(n)
    e = ones(n, 1);
    L = spdiags([e -2*e e], -1:1, n, n);
    L(1, :) = 0; L(1, 1) = -1; L(1, 2) = 1;
    L(n, :) = 0; L(n, n-1) = 1; L(n, n) = -1;
end

function w = stable_reweight(h, opts)
    h = max(h(:), 0);
    hmax = max(h);
    eps_w = max(opts.rew_epsilon_abs, opts.rew_epsilon_rel * max(hmax, eps));
    w = 1 ./ max(h, eps_w);
    w = min(w, opts.rew_wmax);
end

function r_best = select_r_auto(Zp, K, L, opts)
    nG = size(Zp, 1);
    nD = size(K, 2);
    
    idx = randperm(nG);
    nval = max(2, round(opts.r_val_frac * nG));
    id_val = idx(1:nval);
    id_tr = idx(nval+1:end);
    
    if numel(id_tr) < 3
        if opts.verbose
            fprintf('[TRAIn_DOSY_MF] Pocos gradientes para hold-out, usando r=1\n');
        end
        r_best = 1;
        return;
    end
    
    Ztr = Zp(id_tr, :);
    Zva = Zp(id_val, :);
    Ktr = K(id_tr, :);
    Kva = K(id_val, :);
    
    best = inf;
    r_best = 1;
    rmax = max(1, min(opts.r_max, size(Zp, 2)));
    
    for r = 1:rmax
        M = min(opts.init_topM, size(Ztr, 2));
        [~, ord] = sort(sum(Ztr, 1), 'descend');
        sel = ord(1:M);
        
        Hinit = zeros(nD, M);
        for ii = 1:M
            Hinit(:, ii) = lsqnonneg(Ktr, Ztr(:, sel(ii)));
        end
        
        S = init_centroids_farthest(Hinit, r);
        for k = 1:size(S, 2)
            S(:, k) = max(S(:, k), 0);
            S(:, k) = S(:, k) / (sum(S(:, k)) + eps);
        end
        
        B = Ktr * S;
        A = update_A(B, Ztr, opts.lambda_A, opts.A_solver);
        A = max(A, 0);
        A = smooth_A_reflect(A, opts.spectral_smooth_window);
        
        for iter = 1:opts.r_quick_iters
            B = Ktr * S;
            A = update_A(B, Ztr, opts.lambda_A, opts.A_solver);
            A = max(A, 0);
            A = smooth_A_reflect(A, opts.spectral_smooth_window);
            
            Yfit = B * A;
            Res = Ztr - Yfit;
            
            for k = 1:r
                ak = A(k, :);
                den = sum(ak.^2) + eps;
                if den <= 1e-12, continue; end
                Rk = Res + B(:, k) * ak;
                yk = max((Rk * ak.') / den, 0);
                
                h0 = [];
                if opts.warm_start, h0 = S(:, k); end
                
                hk = train_component_traincore(yk, Ktr, L, ...
                    opts.lambda_S_smooth, 0, ones(nD, 1), ...
                    min(150, opts.max_iter), opts.term_factor, h0, opts);
                
                hk = max(hk, 0);
                c = sum(hk) + eps;
                hk = hk / c;
                A(k, :) = A(k, :) * c;
                S(:, k) = hk;
                B(:, k) = Ktr * hk;
            end
        end
        
        Bva = Kva * S;
        Rva = Zva - Bva * A;
        sse = sum(Rva(:).^2);
        n = numel(Zva);
        
        method = lower(string(opts.r_selection_method));
        if method == "cv"
            crit = sse / max(n, 1);
        else
            crit = n * log(sse / n + eps) + opts.r_bic_beta * r * log(n);
        end
        
        if crit < best
            best = crit;
            r_best = r;
        end
    end
    
    if opts.verbose
        fprintf('[TRAIn_DOSY_MF V3.1] auto_r: r=%d (r_max=%d, metodo=%s)\n', ...
            r_best, rmax, char(opts.r_selection_method));
    end
end

function S0 = init_centroids_farthest(H, r)
    H = max(H, 0);
    Hn = H ./ (sum(H, 1) + eps);
    m = size(Hn, 2);
    r = min(r, m);
    cent = zeros(size(Hn, 1), r);
    
    [~, i1] = max(sum(Hn.^2, 1));
    cent(:, 1) = Hn(:, i1);
    
    for k = 2:r
        sims = cent(:, 1:k-1).' * Hn;
        mx = max(sims, [], 1);
        [~, idx] = min(mx);
        cent(:, k) = Hn(:, idx);
    end
    
    labels = ones(1, m);
    for t = 1:3
        sims = cent.' * Hn;
        [~, labels] = max(sims, [], 1);
        for k = 1:r
            sel = (labels == k);
            if any(sel)
                cent(:, k) = mean(Hn(:, sel), 2);
                cent(:, k) = max(cent(:, k), 0);
                cent(:, k) = cent(:, k) / (sum(cent(:, k)) + eps);
            end
        end
    end
    S0 = cent;
end

function S0 = init_nmf_style(H, r)
    H = max(H, 0);
    [nD, m] = size(H);
    
    try
        r_svd = min(r, min(nD, m));
        [U, Sig, ~] = svds(H, r_svd);
    catch
        [U, Sig, ~] = svd(H, 'econ');
        r_svd = min(r, size(U, 2));
        U = U(:, 1:r_svd);
        Sig = Sig(1:r_svd, 1:r_svd);
    end
    
    S0 = max(U * Sig, 0);
    
    r_actual = size(S0, 2);
    if r_actual < r
        S_extra = rand(nD, r - r_actual);
        for k = 1:size(S_extra, 2)
            S_extra(:, k) = S_extra(:, k) / (sum(S_extra(:, k)) + eps);
        end
        S0 = [S0, S_extra];
    end
    
    for k = 1:size(S0, 2)
        S0(:, k) = S0(:, k) / (sum(S0(:, k)) + eps);
    end
end

function S0 = init_centroids_kmeans(H, r)
    H = max(H, 0);
    Hn = H ./ (sum(H, 1) + eps);
    [nD, m] = size(Hn);
    r = min(r, m);
    cent = zeros(nD, r);
    
    [~, i1] = max(sum(Hn.^2, 1));
    cent(:, 1) = Hn(:, i1);
    
    for k = 2:r
        dists = zeros(m, 1);
        for j = 1:m
            min_dist = inf;
            for c = 1:k-1
                d = norm(Hn(:, j) - cent(:, c));
                min_dist = min(min_dist, d);
            end
            dists(j) = min_dist^2;
        end
        probs = dists / (sum(dists) + eps);
        cum_probs = cumsum(probs);
        rnd = rand();
        idx = find(cum_probs >= rnd, 1, 'first');
        if isempty(idx), idx = m; end
        cent(:, k) = Hn(:, idx);
    end
    
    labels = ones(1, m);
    for t = 1:3
        sims = cent.' * Hn;
        [~, labels] = max(sims, [], 1);
        for k = 1:r
            sel = (labels == k);
            if any(sel)
                cent(:, k) = mean(Hn(:, sel), 2);
                cent(:, k) = max(cent(:, k), 0);
                cent(:, k) = cent(:, k) / (sum(cent(:, k)) + eps);
            end
        end
    end
    S0 = cent;
end

function A = update_A(B, Zp, lambdaA, solver)
    [~, r] = size(B);
    
    if strcmpi(solver, 'lsqnonneg')
        A = zeros(r, size(Zp, 2));
        if lambdaA > 0
            Ba = [B; sqrt(lambdaA) * eye(r)];
        else
            Ba = B;
        end
        for f = 1:size(Zp, 2)
            y = Zp(:, f);
            if lambdaA > 0
                ya = [y; zeros(r, 1)];
            else
                ya = y;
            end
            A(:, f) = lsqnonneg(Ba, ya);
        end
    else
        BtB = (B.' * B) + lambdaA * eye(r);
        RHS = B.' * Zp;
        A = BtB \ RHS;
        A = max(A, 0);
    end
end

function A = smooth_A_reflect(A, win)
    if isempty(win) || win <= 1
        A = max(A, 0);
        return;
    end
    
    win = round(win);
    if mod(win, 2) == 0, win = win + 1; end
    pad = floor(win / 2);
    ker = ones(1, win) / win;
    
    for k = 1:size(A, 1)
        a = A(k, :);
        if numel(a) < win, continue; end
        a_pad = [a(pad:-1:1), a, a(end:-1:end-pad+1)];
        a_sm = conv(a_pad, ker, 'same');
        A(k, :) = a_sm(pad+1:pad+numel(a));
    end
    A = max(A, 0);
end

function hk = train_component_traincore(y, K, L, lambdaSmooth, lambdaSparse, w, iterMax, termFac, h0, opts)
    nD = size(K, 2);
    C = K;
    rhs = y(:);
    
    if lambdaSmooth > 0
        C = [C; sqrt(lambdaSmooth) * L];
        rhs = [rhs; zeros(size(L, 1), 1)];
    end
    
    if lambdaSparse > 0
        w = max(w(:), max(opts.rew_epsilon_abs, 1e-12));
        W = spdiags(sqrt(w), 0, nD, nD);
        C = [C; sqrt(lambdaSparse) * W];
        rhs = [rhs; zeros(nD, 1)];
    end
    
    if nargin < 9 || isempty(h0), h0 = []; end
    
    hk = local_train_core_kernel(C, rhs, iterMax, termFac, h0);
end

function R2 = r2_columns(Y, Yhat)
    Y = double(Y); Yhat = double(Yhat);
    SSres = sum((Y - Yhat).^2, 1);
    mu = mean(Y, 1);
    SStot = sum((Y - mu).^2, 1) + eps;
    R2 = 1 - SSres ./ SStot;
    R2 = max(0, min(1, R2));
end

function cons = compute_consistency_metrics(Zp, Yfit, A, S)
    R = Zp - Yfit;
    
    nr = sqrt(sum(R.^2, 1) ./ (sum(Zp.^2, 1) + eps));
    cons.NRMSE_med = median(nr);
    cons.NRMSE_mean = mean(nr);
    
    Zc = Zp - mean(Zp, 1);
    R2 = 1 - sum(R.^2, 1) ./ (sum(Zc.^2, 1) + eps);
    R2 = max(0, min(1, R2));
    cons.R2_med = median(R2);
    cons.R2_mean = mean(R2);
    
    P = A ./ (sum(A, 1) + eps);
    H = -sum(P .* log(P + eps), 1) / log(max(size(A, 1), 2));
    cons.entropy_med = median(H);
    cons.entropy_mean = mean(H);
    
    if size(A, 2) > 1
        cons.A_TV = mean(abs(diff(A, 1, 2)), 'all') / (mean(A(:)) + eps);
    else
        cons.A_TV = 0;
    end
    
    r = size(S, 2);
    if r > 1
        Sn = S ./ (sqrt(sum(S.^2, 1)) + eps);
        sim_matrix = Sn.' * Sn;
        off_diag = sim_matrix - diag(diag(sim_matrix));
        cons.component_separation = 1 - mean(abs(off_diag(:)));
    else
        cons.component_separation = 1;
    end
    
    score = 0.40 * cons.R2_med + ...
            0.20 * (1 - cons.NRMSE_med) + ...
            0.15 * (1 - cons.entropy_med) + ...
            0.25 * cons.component_separation;
    cons.score = max(0, min(1, score));
end

function [nSub, subIdx] = subsample_indices(n, maxN)
    if n <= maxN
        subIdx = 1:n;
        nSub = n;
        return;
    end
    subIdx = round(linspace(1, n, maxN));
    subIdx = unique(subIdx);
    nSub = numel(subIdx);
end

function Sim = cosine_similarity_cols(A)
    A = double(A);
    norms = sqrt(sum(A.^2, 1)) + eps;
    An = A ./ norms;
    Sim = An.' * An;
    Sim = max(0, min(1, Sim));
end

function h = local_train_core_kernel(C, s, iterMax, termFac, h0)
    s = s(:);
    [~, n] = size(C);
    
    if any(~isfinite(s)) || all(s <= eps)
        h = zeros(n, 1);
        return;
    end
    
    hNNLS = lsqnonneg(C, s);
    noisLev = sqrt((C * hNNLS - s)' * (C * hNNLS - s));
    if ~(isfinite(noisLev) && noisLev > 0)
        noisLev = max(1e-12, norm(s));
    end
    
    if nargin >= 5 && ~isempty(h0)
        h0 = max(h0(:), 0);
        if numel(h0) ~= n, h0 = []; end
    else
        h0 = [];
    end
    
    if ~isempty(h0) && any(h0 > 0)
        h = max(h0, 1e-12 * max(h0));
    else
        minS = min(abs(s));
        if ~(isfinite(minS) && minS > 0), minS = eps; end
        h = 1e-10 * ones(n, 1) * minS / n;
    end
    
    eta = sqrt(h);
    
    subMax = 100;
    radius0 = 0.01 * sqrt(sqrt(s' * s));
    radius = max(radius0, 1e-16);
    
    thr0 = 0.01; thr1 = 0.25; thr2 = 0.75;
    mul1 = 0.5; mul2 = 2;
    
    CTC = C' * C;
    CTs = C' * s;
    
    f = (C * h - s)' * (C * h - s);
    iter = 0;
    exitMain = false;
    
    hess_mult = @(v, eta_) 8 * (eta_ .* (CTC * (eta_ .* v)));
    
    while iter < iterMax && ~exitMain
        iter = iter + 1;
        
        g0 = (CTC * h - CTs);
        grad = 4 * (eta .* g0);
        
        sSub = zeros(n, 1);
        gSub = grad;
        dSub = -gSub;
        
        iterSub = 0;
        exitSub = false;
        
        while iterSub < subMax && ~exitSub
            iterSub = iterSub + 1;
            
            Hd = hess_mult(dSub, eta);
            kappa = dSub' * Hd;
            
            if kappa <= 0
                exitSub = true;
            else
                alpha = -(gSub' * dSub) / (kappa + eps);
                sSub1 = sSub + alpha * dSub;
                
                if (sSub1' * sSub1) >= radius^2
                    exitSub = true;
                else
                    gSub1 = gSub + alpha * Hd;
                    beta = (gSub1' * gSub1) / (gSub' * gSub + eps);
                    dSub = -gSub1 + beta * dSub;
                    sSub = sSub1;
                    gSub = gSub1;
                end
            end
            
            if exitSub
                disc = (sSub' * dSub)^2 - (dSub' * dSub) * (sSub' * sSub - radius^2);
                if disc >= 0
                    lambda = (sqrt(disc) - sSub' * dSub) / (dSub' * dSub + eps);
                    sSub = sSub + lambda * dSub;
                end
            end
        end
        
        eta1 = abs(eta + sSub);
        h1 = eta1.^2;
        f1 = (C * h1 - s)' * (C * h1 - s);
        
        Hs = hess_mult(sSub, eta);
        denom = (grad' * sSub + 0.5 * (sSub' * Hs));
        if abs(denom) < eps
            rho = -Inf;
        else
            rho = (f1 - f) / denom;
        end
        
        if rho > thr0
            eta = eta1;
            h = h1;
            f = f1;
        end
        
        if rho <= thr1
            radius = max(mul1 * radius, 1e-12 * radius0);
        elseif rho >= thr2
            radius = min(mul2 * radius, radius0);
        end
        
        if sqrt(max(f, 0)) <= termFac * noisLev
            exitMain = true;
        end
    end
    
    mx = max(h);
    if mx > 0
        h(h < 1e-6 * mx) = 0;
    end
end