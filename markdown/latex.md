
My Latex Note
======================
####实用Latex技能收集

##### 1. 在线latex编辑器推荐(送给像我一样嫌本地编译麻烦的懒人们)
> +  **Overleaf**: [https://www.overleaf.com](https://www.overleaf.com) 有很多模板, 与许多会议,期刊都有合作,对中文支持一般,速度稍慢
> +  **ShareLatex**: [https://www.sharelatex.com/](https://www.sharelatex.com/) 对中文支持的比较好,个人的本科毕业设计就是用这个写的,速度还可以
> + **CloudLatex**: [https://cloudlatex.io/](https://cloudlatex.io/) 日本人写的一个网站, 个人感觉效果一般

##### 2. 常用latex语法
> + **文献引用**: \citep\{socher13\}, \citet\{socher13\},\citealt\{socher13\}等, 根据效果选择
> + **数学公式**: 
	> 行内公式: \$ balabalabala \$ 
	> 行间公式: 
		 <pre>\begin{equation}
\label{eqn:01} % label 表示公式编号
P_{\Sigma}, P_{\Omega} = CCA(\Sigma',\Omega')
\end{equation} </pre>
> + **数学公式相关符号**:
	> **$\mathbb{R}$**:  \mathbb{R}
	> **$\in$**: \in
	> **$x_1 \times y_1$** :  x_1 \times y_1
	> **$x_1^2 + y_1^2$**: x_1^2 + y_1^2
> + **各类数学符号**
> 1. 数学模式重音符号 
> **$\hat{a}$**:  \hat{a}　　　　 **$\check{a}$**: \cheack{a}　　　　**$\tilde{a}$**: \tilde{a}
> **$\grave{a}$**:  \grave{a}　　　　 **$\dot{a}$**: \dot{a}　　　　**$\ddot{a}$**: \ddot{a}
> **$\bar{a}$**:  \bar{a}　　　　 **$\vec{a}$**: \vec{a}　　　　**$\widehat{a}$**: \widehat{a}
> **$\acute{a}$**:  \acute{a}　　　　 **$\breve{a}$**: \breve{a}　　　　**$\widetilde{a}$**: \widetilde{a}
>  2. 希腊字母
>  **$\alpha$**:  \alpha　　　　 **$\theta$**: \theta　　　　**$o$**: o　　　　**$\upsilon$**: \upsilon
>  **$\beta$**:  \beta　　　　 **$\vartheta$**: \vartheta　　　　**$\pi$**: \pi　　　　**$\phi$**: \phi
>  **$\gamma$**:  \gamma　　　　 **$\iota$**: \iota　　　　**$\varpi$**: \varpi　　　　**$\varphi$**: \varphi
>  **$\delta$**:  \delta　　　　 **$\kappa$**: \kappa　　　　**$\rho$**: /rho　　　　**$\chi$**: \chi
>  **$\epsilon$**:  \epsilon　　　　 **$\lambda$**: \lambda　　　　**$\varrho$**: \varrho　　　　**$\psi$**: \psi
>  **$\varepsilon$**:  \varepsilon　　　　 **$\mu$**: \mu　　　　**$\sigma$**: \sigma　　　　**$\omega$**: \omega
>  **$\zeta$**:  \zeta　　　　 **$\nu$**: \nu　　　　**$\varsigma$**: \varsigma　　　　
>  **$\eta$**:  \eta　　　　 **$\xi$**: \xi　　　　**$\tau$**: \tau　　　　
>  **$\Gamma$**:  \Gamma　　　　 **$\Lambda$**: \Lambda　　　　**$\Sigma$**: \Sigma　　　　**$\Psi$**: \Psi
>  **$\Delta$**:  \Delta　　　　 **$\Xi$**: \Xi　　　　**$\Upsilon$**: \Upsilon　　　　**$\Omega$**: \Omega
>  **$\Theta$**:  \Theta　　　　 **$\Pi$**: \Pi　　　　**$\Phi$**: \Phi
>  3. 二元运算符
>  **$\pm$**:  \pm　　　　 **$\mp$**: \mp　　　　**$\triangleleft$**: \triangleleft
>  **$\cdot$**:  \cdot　　　　 **$\div$**: \div　　　　**$\triangleright$**: \triangleright
>  **$\times$**:  \times　　　**$\setminus$**: \setminus　　**$\star$**: \star
>        **$\cup$**:  \cup　　　　 **$\cap$**: \cap　　　　**$\ast$**: \
>        **$\vee$**:  \vee　　　　 **$\wedge$**: \wedge　　   　**$\bullet$**: \bullet
>            **$\otimes$**:  \otimes　　　**$\oplus$**: \oplus　　　**$\odot$**: \odot
> 4. 二元关系
>   **$\le$**:  \le　　　　 **$\ge$**: \ge　　　　**$\equiv$**: \equiv
>   **$\ll$**:  \ll　　　　 **$\gg$**: \gg　　　　**$\doteq$**: \doteq
>   **$\sim$**:  \sim　　　 **$\simeq$**: \simeq　  　 **$\subset$**: \subset
>   **$\supset$**:  \supset　　**$\approx$**: \approx　　 **$\cong$**: \cong
>   **$\in$**:  \in　　　　 **$\Join$**: \Join　　　 **$\bowtie$**: \bowtie
>   **$\propto$**:  \propto　　**$\mid$**: \mid　　　　 **$\parallel$**: \parallel
>   **$\notin$**:  \notin　　 **$\ne$**: \
>    5. 大运算符
>    **$\sum$**:  \sum　　　 **$\bigcup$**: \bigcup　　 **$\bigvee$**: \bigvee
>    **$\prod$**:  \prod　　　 **$\bigcap$**: \bigcap　　 **$\bigwedge$**: \bigwedge
>    **$\int$**:  \int　　　 **$\oint$**: \oint　　 **$\bigodot$**: \bigodot
>    **$\bigoplus$**:  \bigoplus　　　 **$\bigotimes$**: \bigotimes　　 
>    6. 箭头类
>    **$\leftarrow$**:  \leftarrow　　　 **$\rightarrow$**: \leftarrow
>    **$\longleftarrow$**:  \longleftarrow　　　 **$\longrightarrow$**: \longleftarrow
>    **$\leftrightarrow$**:  \leftrightarrow　　　 **$\longleftrightarrow$**: \longleftrightarrow
>    **$\Leftarrow$**:  \Leftarrow　　　 **$\Rightarrow$**: \Rightarrow
>    **$\Longleftarrow$**:  \Longleftarrow　　　 **$\Longrightarrow$**: \Longrightarrow　　 
>    **$\Leftrightarrow$**:  \Leftrightarrow　　　 **$\Longleftrightarrow$**: \Longleftrightarrow 
>    **$\rightleftharpoons$**:  \rightleftharpoons　　　 
>    **$\uparrow$**:  \uparrow　　　 **$\downarrow$**: \downarrow
>    **$\Uparrow$**:  \Uparrow　　　 **$\Downarrow$**: \Downarrow
>    **$\updownarrow$**:  \updownarrow　　　 **$\Updownarrow$**: \Updownarrow
>    7. 其他符号
>    **$\dots$** :  \dots　　　　 **$\cdots$** :  \cdots　　　　**$\vdots$** : \vdots　　　　**$\ddots$** :  \ddots
>    **$\hbar$** :  \hbar　　　　 **$\imath$** :  \imath　　　　**$\jmath$** : \jmath　　　　**$\ell$** :  \ell
>    **$\forall$** :  \forall　　　　 **$\exists$** :  \exists　　　　**$\partial$** : \partial　　　　**$\emptyset$** :  \emptyset
>   **$\infty$** :  \infty　　　　 **$\nabla$** :  \nabla　　　　**$\angle$** : \angle　　　　**$\surd$** :  \surd
>   8. 非数学符号(文本模式里常用的)
>   **$\S$** :  \S　　　　 **$\copyright$** :  \copyright　　　　**$\textregistered$** : \textregistered　　　　**$\%$** :  \%   
+ 其他内容
> 1. 图片
> <pre>\begin{figure}
\includegraphics[width=0.5\textwidth,height=0.3\textwidth]{filename}
\caption{ balabalabala}		
\label{fig:two}		%编号
\end{figure}</pre>
> 2. 表格
> <pre>\begin{tabular}{|c|c|c|}
	\hline 2&9&4\\
	\hline 7&5&3\\
	\hline 6&1&8\\
	\hline
\end{tabular}</pre>
> 3. footnote 
> \footnote{balabala}