%loading data
load data;

[W,A] = fastica(artificial,2);
fprintf('W_1 = %d\n',var(W(1,:)*artificial));
fprintf('W_2 = %d\n',var(W(2,:)*artificial));

plot(W(1,:)*artificial);
figure;
plot(W(2,:)*artificial);

component = 1;
data = audio2;
[W_audio,A_audio] = fastica(data,2);
% fprintf('W_audio_1 = %d\n',var(W_audio(1,:)*data));
% fprintf('W_audio_2 = %d\n',var(W_audio(2,:)*data));

mixFlag = 2;

if mixFlag == 1
    fprintf('playing mixed audio\n');
    soundsc(data(component,:), 44100);
else
    fprintf('playing unmixed audio\n');
    soundsc(W_audio(2,:)*data, 44100);
end

fprintf('patch\n');
[W_patch, A_patch] = fastica(patches,30);
figure;
displaycolumns(A_patch);