# 数学附录

## A. 分形测度与积分
分形 $\mathcal{F}_D$ 上的积分定义为：
$$\int_{\mathcal{F}_D} f(x) d\mu_D(x) = \lim_{n\to\infty} \sum_{\alpha} f(x_\alpha) \mu_D(Q_\alpha)$$

## B. 分形上的RG流
拓扑角的β函数：
$$\beta_\theta^{(\mathcal{F})} = -\frac{D}{4}\frac{y^2 g_A}{\pi^4}\theta + \frac{g_A^3}{3\pi^2}\frac{\Gamma(D/2)}{\Gamma(2)} + \Delta\beta$$

## C. 数值算法
自洽求解采用不动点迭代，收敛条件 $|\theta_{new} - \theta_{old}| &lt; 10^{-6}$