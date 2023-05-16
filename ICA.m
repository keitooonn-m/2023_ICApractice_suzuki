clear; clc; close all;

% 音源読み込み
[sig_s1, Fs] = audioread("audiofiles\audio1.wav"); %#ok<ASGLU> 
[sig_s2, Fs] = audioread("audiofiles\audio2.wav");

% 信号の長さを長いほうに揃える
if length(sig_s1) ~= length(sig_s2)
    [sig_s1, sig_s2, sig_len] = AlignSignal(sig_s1, sig_s2);
end
S = [sig_s1, sig_s2];

% 混合音を疑似的に収録(もともと混合されている場合はスルー)
A = [0.6, 0.7;
     0.8, 0.5];
[X, x1, x2] = MixSound(S, A);
% 混合されている場合は以下を実行
% x1 = sig_s1;
% x2 = sig_s2;
% X = [x1, x2];

% 自然勾配法でBSS
step_size = 0.5;
replay = 50;
random_seed = 200;
[Y, W, costfunc_history] = BSS(X, step_size, replay, random_seed, sig_len);
plot(costfunc_history)

%プロジェクションバック
[audio_y1, audio_y2] = ProjectionBack(Y, W);

% 音声出力
audiowrite("audiofiles\mixedaudio1.wav",x1,Fs)
audiowrite("audiofiles\mixedaudio2.wav",x2,Fs)
audiowrite("audiofiles\separatedaudio1.wav",audio_y1(1,:),Fs)
audiowrite("audiofiles\separatedaudio2.wav",audio_y2(1,:),Fs)

%--------------------------------------------------------------------
function [sig_s1, sig_s2, sig_len] = AlignSignal(sig_s1, sig_s2)
% 信号長をそろえる
%
% @return sig_s1 統一された長さの元音源1
% @return sig_s1 統一された長さの元音源2
% @return sig_len 統一された音源の長さ
% @param sig_s1 元音源1
% @param sig_s2 元音源2
    s1_len = length(sig_s1);
    s2_len = length(sig_s2);
    if s1_len >= s2_len
        sig_s2(end+1:s1_len) = 0;
    else
        sig_s1(end+1:s1_len) = 0;
    end
    sig_len = length(sig_s1);
end

function [X, x1, x2] = MixSound(S, A)
% 疑似的な音源の混合
%
% @return X 混合音源の行列
% @return x1 混合音源1
% @return x2 混合音源2
% @param S 元音源の行列
% @param A 混合行列
    X = A * S';
    x1 = X(1,:);
    x2 = X(2,:);
end

function [Y, W, costfunc_history] = BSS(X, step_size, replay, random_seed, sig_len)
% BSSを行う
%
% @return Y 分離音源の行列
% @return W 推測した混合行列
% @return costfunc_history 推測した混合行列の評価値
% @param X 混合音源の行列
% @param step_size ステップサイズ
% @param replay 最急降下法の反復回数
% @param random_seed ランダムシード値
% @param sig_len 信号長 
    rng(random_seed);
    W = randn(2);
    I = eye(2);
    E = zeros(2);
    costfunc_history = zeros(1, replay);
    % 生成モデル
    PFunc = @(Y) sech(Y)/pi;
    % スコア関数
    PhiFunc = @(Y) tanh(Y);
    % KLダイバージェンス
    KLdivFunc = @(W, Y) -log(abs(det(W))) - sum(log(PFunc(Y)), 'all')/sig_len; 
    
    for l = 0:replay-1
        % 期待値計算
        E = (PhiFunc(W*X) * (W*X)')/sig_len;
        % 最急降下法
        W = W - step_size * (E-I) * W;
        % コスト関数計算
        Y = W * X;
        costfunc_history(l+1) = KLdivFunc(W, Y);
    end
end

function [audio_y1, audio_y2] = ProjectionBack(Y, W)
% プロジェクションバックを行う
%
% @return audio_y1 プロジェクションバック後の分離音源1
% @return audio_y2 プロジェクションバック後の分離音源2
% @param Y 分離音源の行列
% @param W 推測した混合行列
    y1(1,:) = Y(1,:);
    y2(2,:) = Y(2,:);
    y1(2,:) = 0;
    y2(1,:) = 0;
    audio_y1 = W \ y1;
    audio_y2 = W \ y2;
end