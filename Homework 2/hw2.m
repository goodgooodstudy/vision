%loading data
load data;

[W,A] = fastica(artificial,2);
fprintf('W_1 = %d\n',var(W(1,:)*artificial));
fprintf('W_2 = %d\n',var(W(2,:)*artificial));

plot(W(1,:)*artificial);
figure;
plot(W(2,:)*artificial);