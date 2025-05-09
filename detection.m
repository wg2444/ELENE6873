clear; clc; close all;

%% 1. 参数设置
N         = 128;                 % 信号长度
Fs        = 1000;                % 采样率 (Hz)
A         = 1;                   % 信号幅度
numTrials = 3000;                % Monte‑Carlo 次数
SNR_dB    = -10:5:10;            % SNR 范围 (dB)
rho       = 0.8;                 % AR(1) 系数
Pfa_req   = 0.05;                % 固定误警率
numThresh = 200;                 % ROC 阈值点数

%% 2. 模板信号 (列向量 + 单位能量归一化)
n       = (0:N-1)';
s_pulse = hann(N);    s_pulse = s_pulse / norm(s_pulse);
s_chirp = chirp(n/Fs,50,N/Fs,200); s_chirp = s_chirp / norm(s_chirp);
templates     = {s_pulse, s_chirp};
templateNames = {'Pulse','Chirp'};

%% 3. AR(1) 白化滤波器系数
b = [1 -rho];
a = 1;

%% 4. 结果预分配
numTmpl = numel(templates);
numSNR  = numel(SNR_dB);
results = repmat(struct( ...
    'Fraw',[],'Traw',[],'AUCRaw',[],'PDraw',[], ...
    'Fwh',[] ,'Twh',[] ,'AUCWh',[] ,'PDWh',[]), ...
    numTmpl, 2, numSNR);

%% 5. Monte‑Carlo 仿真
for ti = 1:numTmpl
    s_raw = templates{ti}(:);
    % 白化后模板
    s_wh  = filter(b,a,s_raw);
    s_wh  = s_wh / norm(s_wh);
    
    for si = 1:numSNR
        sigma2 = A^2 / 10^(SNR_dB(si)/10);
        stat_raw0 = zeros(numTrials,1);
        stat_raw1 = zeros(numTrials,1);
        stat_wh0  = zeros(numTrials,1);
        stat_wh1  = zeros(numTrials,1);
        
        for k = 1:numTrials
            % 产生 AR(1) 有色噪声
            w     = randn(N,1);
            n_col = filter(1,[1 -rho], sqrt(sigma2)*w);
            % 噪声白化
            n_wh  = filter(b,a,n_col);
            
            % 构造 H0/H1 接收向量
            y0_raw = n_col;
            y1_raw = A*s_raw + n_col;
            y0_wh  = n_wh;
            y1_wh  = A*s_wh  + n_wh;
            
            % 计算检测统计量 (Matched Filter)
            stat_raw0(k) = dot(s_raw, y0_raw);
            stat_raw1(k) = dot(s_raw, y1_raw);
            stat_wh0(k)  = dot(s_wh,  y0_wh);
            stat_wh1(k)  = dot(s_wh,  y1_wh);
        end
        
        % 原始 MF ROC & AUC & PD@PFA
        [Fraw,Traw] = simpleROC(stat_raw0, stat_raw1, numThresh);
        AUCRaw      = abs(trapz(Fraw, Traw));
        thr_raw     = quantile(stat_raw0, 1-Pfa_req);
        PDraw       = mean(stat_raw1 > thr_raw);
        
        % 白化后 MF ROC & AUC & PD@PFA
        [Fwh,Twh] = simpleROC(stat_wh0, stat_wh1, numThresh);
        AUCWh     = abs(trapz(Fwh, Twh));
        thr_wh    = quantile(stat_wh0, 1-Pfa_req);
        PDWh      = mean(stat_wh1 > thr_wh);
        
        % 存储结果
        results(ti,1,si) = struct('Fraw',Fraw,'Traw',Traw, 'AUCRaw',AUCRaw,'PDraw',PDraw, ...
                                   'Fwh',[]  ,'Twh',[]   ,'AUCWh',[]    ,'PDWh',[]);
        results(ti,2,si) = struct('Fraw',[]  ,'Traw',[]  , 'AUCRaw',[]   ,'PDraw',[],   ...
                                   'Fwh',Fwh ,'Twh',Twh ,'AUCWh',AUCWh  ,'PDWh',PDWh);
    end
end

%% 6. 绘图
% (1) ROC: Pulse, Raw vs Whitened
%% Pulse ROC：2×3 正方形 + 常规 legend
figure('Name','ROC: Pulse Raw vs Whitened', ...
       'NumberTitle','off', ...
       'Units','normalized', ...
       'Position',[0.1, 0.1, 0.4, 0.6]);   % 正方形窗口
tiledlayout(3,2,'Padding','tight','TileSpacing','compact');

hRaw = []; hWh = [];
for si = 1:numSNR
    ax = nexttile;
    hold(ax,'on');
    % 原始样式绘线
    h1 = plot(ax, results(1,1,si).Fraw, results(1,1,si).Traw);
    h2 = plot(ax, results(1,2,si).Fwh,  results(1,2,si).Twh);
    % 参考线
    plot(ax, [0 1],[0 1],'k--');
    xlabel(ax,'P_{FA}');
    ylabel(ax,'P_D');
    title(ax,sprintf('SNR = %d dB', SNR_dB(si)));
    axis(ax,[0 1 0 1]); grid(ax,'on');
    hold(ax,'off');
    if si == 1
        hRaw = h1; hWh = h2;  % 只取第一个子图的线条句柄用于图例
    end
end
% 如果只有 5 个 SNR，就隐藏第 6 个空位
if numSNR < 6
    ax = nexttile(6); axis(ax,'off');
end
% 常规 legend，不指定 tiledlayout
legend([hRaw hWh], {'Raw MF','Whitened MF'}, ...
       'Orientation','horizontal', 'Location','southoutside');


% (3) ROC: Chirp, Raw vs Whitened
%% Chirp ROC：2×3 正方形 + 常规 legend
figure('Name','ROC: Chirp Raw vs Whitened', ...
       'NumberTitle','off', ...
       'Units','normalized', ...
       'Position',[0.1, 0.1, 0.4, 0.6]);  
tiledlayout(3,2,'Padding','tight','TileSpacing','compact');

hRaw = []; hWh = [];
for si = 1:numSNR
    ax = nexttile;
    hold(ax,'on');
    h1 = plot(ax, results(2,1,si).Fraw, results(2,1,si).Traw);
    h2 = plot(ax, results(2,2,si).Fwh,  results(2,2,si).Twh);
    plot(ax, [0 1],[0 1],'k--');
    xlabel(ax,'P_{FA}');
    ylabel(ax,'P_D');
    title(ax,sprintf('SNR = %d dB', SNR_dB(si)));
    axis(ax,[0 1 0 1]); grid(ax,'on');
    hold(ax,'off');
    if si == 1
        hRaw = h1; hWh = h2;
    end
end
if numSNR < 6
    ax = nexttile(6); axis(ax,'off');
end
legend([hRaw hWh], {'Raw MF','Whitened MF'}, ...
       'Orientation','horizontal', 'Location','southoutside');


% (2) AUC vs SNR
figure('Name','AUC vs SNR','NumberTitle','off'); hold on;
markers = {'o','s'};
for ti = 1:numTmpl
    auc_raw = arrayfun(@(k) results(ti,1,k).AUCRaw, 1:numSNR);
    auc_wh  = arrayfun(@(k) results(ti,2,k).AUCWh,  1:numSNR);
    plot(SNR_dB, auc_raw, ['-' markers{ti}], 'DisplayName',[templateNames{ti} ' Raw']);
    plot(SNR_dB, auc_wh,  ['--' markers{ti}], 'DisplayName',[templateNames{ti} ' Whitened']);
end
xlabel('SNR (dB)'); ylabel('AUC'); legend('Location','best'); grid on;

%% 辅助函数
function [FPR,TPR] = simpleROC(s0,s1,numT)
    scores = [s0; s1];
    thr    = linspace(min(scores),max(scores),numT);
    FPR    = zeros(1,numT);
    TPR    = zeros(1,numT);
    for i = 1:numT
        FPR(i) = mean(s0 > thr(i));
        TPR(i) = mean(s1 > thr(i));
    end
end