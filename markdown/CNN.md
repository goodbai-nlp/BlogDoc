<h2 id="convolutional-neural-networks">Convolutional Neural Networks</h2>

<blockquote>
  <p>白雪峰 — xfbai@mtlab.hit.edu.cn.</p>
</blockquote>



<h2 id="outline">Outline</h2>

<ul>
<li>CNN栗子镇楼</li>
<li>What is CNN <br>
<ul><li>什么是卷积</li>
<li>什么是池化</li></ul></li>
<li>Why CNN </li>
<li>对CNN的其他一些理解</li>
<li>CNN实现（接口）</li>
</ul>



<h2 id="1-cnn栗子a-beginning-glimpse-of-cnn">1. CNN栗子（A Beginning Glimpse of CNN）</h2>

<p>(1)Modern CNN since Yann LeCun <br>
<img src="https://lh3.googleusercontent.com/IoYo64NskyHdxzltw-xEk5abHX1_Lmk8uTziFrd4S8C6fZdzdoMDd_x1FOXQiV9CQ4VzuC_xZNQ=s0" alt="enter image description here" title="lecun.png"> <br>
(2) <br>
<img src="https://lh3.googleusercontent.com/i5NTRpYY7AGaDoLpnkKEtvVdCKYkEYswIBdlthKLjAyjp2uZtQ1f_pIqFXqI5dtxUAG4hT3Qq2E=s0" alt="DeepID" title="DeepId.png"></p>



<h2 id="2-what-is-cnn">2. What is CNN?</h2>

<p>神经网络？卷积？</p>



<h3 id="21-什么是卷积">2.1 什么是卷积？</h3>



<h4 id="卷积的定义">卷积的定义:</h4>

<p>其连续的定义为： <br>
<script type="math/tex" id="MathJax-Element-1">(f*g)(n) = \int_{ - \infty }^{ + \infty } {f(t)g(n-t)dt}</script> <br>
其离散的定义为： <br>
 <script type="math/tex" id="MathJax-Element-2">(f*g)(n) = \sum_{t = - \infty} ^{ + \infty } {f(t)g(n-t)}</script></p>

<p>特点：　 <br>
<img src="https://lh3.googleusercontent.com/kHgn5WtCxzUHBbh9GR-EoW0WB1NS5mbRzAGT4LpbwqrvACwDERgXIRaecJ4E4a9JiLavTA8GLKs=s0" alt="enter image description here" title="卷积1.jpg"></p>



<h3 id="22-离散卷积的栗子">2.2 离散卷积的栗子：</h3>

<p>丢骰子，两个骰子加起来要等于4的概率是多少？</p>

<blockquote>
  <p><script type="math/tex" id="MathJax-Element-3">(f*g)(4) = \sum_{m = 1} ^{ 3 } {f(m)g(4-m)}</script></p>
</blockquote>



<h4 id="a-二维离散的卷积">a. 二维离散的卷积：</h4>

<p><script type="math/tex" id="MathJax-Element-4">(f*g)(m,n) = \sum\limits_{k=0}^{2}\sum\limits_{h=0}^{2}f(h,k)g(m-h,n-k)</script> <br>
则 (f*g)(1,1) ? <br>
<img src="https://lh3.googleusercontent.com/tbQsGAO_VeDVH4mtHlrY4F43alqwR2KrMbNfFuXOVEqmebEo_qmrwvPhNVZ4XCJOebz5Bv6Ujvk=s0" alt="enter image description here" title="111.gif"></p>

<blockquote>
  <p>离散的卷积可以看成是矩阵的乘法（翻转的）</p>
</blockquote>



<h3 id="23-用到二维图像上">2.3 用到二维图像上：</h3>



<h4 id="a关于卷积中常用到的一些概念">a.关于卷积中常用到的一些概念：</h4>

<blockquote>
  <p>In <strong>Image Processing</strong> <br>
  – Convolution is always named <strong>filtering</strong>, and there are many famous <strong>filters/convolution kernels</strong> that <strong><em>extract intuitive features in images</em></strong></p>
</blockquote>

<p><img src="https://lh3.googleusercontent.com/OVoi8qCglOwEJ0OaJPT8cXZW10fj37-eGCSZLf81m33NF16pFg3kYsbz0rvAzvLYYer-9yD7Rks=s0" alt="enter image description here" title="222.gif"> <br>
通过在输入数据上不断移动卷积核，来提取不同位置的特征．</p>



<h4 id="b-图像上作卷积的效果">b. 图像上作卷积的效果：</h4>

<p><img src="https://lh3.googleusercontent.com/iCsuVDfVvDIRU5qvsUVfs14EV6ZL2y9joMy379_0HjfbWJEu1ieB0nwdv4vaysWSdiwZojlxUQc=s0" alt="enter image description here" title="a.png"> <br>
<img src="https://lh3.googleusercontent.com/Cz0OC3027jUDRbR3M_MxUVEBgdP1hHXNkLZxQDrvfK6G55if9KB_of_Puqj5OCYm6zGmVgXJcnY=s0" alt="enter image description here" title="b.png"></p>



<h3 id="24-用到神经网络中">2.4 用到神经网络中</h3>

<p><img src="https://lh3.googleusercontent.com/bOXd_J8N3dttwdyGUZpeoNkCX5GXHJdNe0QXdQfvGdad46Ue51n0LwE_vqtZLM1bqLtuvrV3aCs=s0" alt="enter image description here" title="c.png"></p>



<h3 id="25-卷积的细节">2.5 卷积的细节</h3>



<h4 id="a-filterkernel-size-number">a. <strong>filter/Kernel size, number</strong></h4>

<p>假设神经网络的输入是６＊６的image, <br>
<img src="https://lh3.googleusercontent.com/d-JO3HlQ84n3J3ezH3yk7CxxdMkrZwwbMQaSWPFH2G0WyLxHbDXk5x4T7g6Zc1IhbA8w_jsI5CI=s0" alt="enter image description here" title="e.png"> <br>
那么，，，</p>

<p>再来个更形象的： <br>
<img src="https://lh3.googleusercontent.com/DSd7g_WeKsPbijUx3lQ4GUsvne5AmBrCV_bodA55Rvy2T4cY3S8pi2oSvNehjx7xIHy03aSsq9g=s0" alt="enter image description here" title="f.png"></p>



<h4 id="b-stride"><strong>b. Stride</strong></h4>

<blockquote>
  <p>The step size you take the filter to sweep the <br>
  image</p>
</blockquote>

<p><img src="https://lh3.googleusercontent.com/-5Yz8fp7qtAlITJQ0tvqi_Are9FaWZRI9JqkywY-0Uqh_zI9NAgM01UxRK1F9XN5EDuuu422ees=s0" alt="enter image description here" title="g.png"></p>



<h4 id="c-zero-padding"><strong>c. Zero-padding</strong></h4>

<blockquote>
  <ol>
  <li>A way not to ignore pattern on border</li>
  <li>New image is smaller than the original image</li>
  </ol>
</blockquote>

<p><img src="https://lh3.googleusercontent.com/z9QvwT3qvYWooGsSquyQkVZ_-UzQg7vyMfcru1JE0V8wPkauMLglRMEVW_HbLiLDrO4zJlEF1QA=s0" alt="enter image description here" title="i.png"></p>



<h4 id="dchannel"><strong>d.Channel</strong></h4>

<p><img src="https://lh3.googleusercontent.com/qiKdcJQ2RW9ReMczWCjPoVe8xHZMOdBUes_9oJ9e5syUvVmLVdyZ5z_2dPEf4vMBufdmwTN2wlY=s0" alt="enter image description here" title="j.png"></p>



<h3 id="26-池化pooling">2.6 池化(Pooling)</h3>

<blockquote>
  <p>Pooling layers <strong>subsample</strong> their input <br>
       1. Spatial pooling (also called subsampling or <br>
  downsampling) <strong><em>reduces the dimensionality</em></strong> of <br>
  each feature map <br>
      2. Retains the <strong><em>most important information</em></strong></p>
</blockquote>



<h4 id="1-max-pooling例子">1. Max pooling例子：</h4>

<p><img src="https://lh3.googleusercontent.com/Xi65Dgab6okAbsBrEWptHgdDdVD8YVw4h8J8Xryvh7UTccLti6dbtPzBFVhuSSpSrfScI6EQ9EA=s0" alt="enter image description here" title="k.png"></p>

<blockquote>
  <p>Pooling is <strong>unsensitive to local translation.</strong>(局部不变性) <br>
  – “if we translate the input by a small amount, <br>
  the values of most of the pooled outputs do <br>
  not change.” <br>
  <img src="https://lh3.googleusercontent.com/-6QNd1M91W90/WivaZhBVFoI/AAAAAAAAAD0/oiIGDrhrzDcnxDsNYSdDeyTPd10lRxzzACLcBGAs/s0/l.png" alt="enter image description here" title="l.png"></p>
</blockquote>



<h4 id="2-pooling的变种">2. Pooling的变种</h4>

<p>Pooling should be designed to fit specific <br>
applications.</p>

<ul>
<li>Max pooling</li>
<li>Average pooling</li>
<li>Min pooling</li>
<li>l 2 -norm pooling</li>
<li>Dynamic k-pooling</li>
<li>Etc.</li>
</ul>



<h4 id="3-pooling的性质">3. Pooling的性质</h4>

<ul>
<li>Makes the input representation (feature dim) smaller and more manageable</li>
<li>Reduces number of parameters and computations in the network, therefore, controlling overfitting</li>
<li>Makes the network invariant to small transformations, distortions and translations in the input image</li>
<li>Help us arrive at almost scale invariant representation of our image</li>
</ul>



<h3 id="27-flatten">2.7 Flatten</h3>

<p><img src="https://lh3.googleusercontent.com/-lE-sEVVpfMM/WivcIGABLII/AAAAAAAAAEM/eDZ0M_s83O0wYgA0tCLqUq2DeZtDNI7dACLcBGAs/s0/m.png" alt="enter image description here" title="m.png"></p>



<h3 id="28-convolution-vs-fully-connected">2.8 Convolution v.s. Fully Connected</h3>

<p><img src="https://lh3.googleusercontent.com/-rYDsoOFK74o/WivdNg46dsI/AAAAAAAAAEc/-jrkYV_J6Og3Enz3sz89-eiWSaOVfQrEQCLcBGAs/s0/n.png" alt="enter image description here" title="n.png"></p>

<p><img src="https://lh3.googleusercontent.com/-QW2ntGUCi8Q/WivdgjYoy5I/AAAAAAAAAEk/Y8Wnp92JzDgKsJPbsX9UinR8s8qCaiboQCLcBGAs/s0/o.png" alt="enter image description here" title="o.png"></p>

<p><img src="https://lh3.googleusercontent.com/-k5Z2j5gi4Ps/WivdryPzHgI/AAAAAAAAAEs/kmQJiYec-PQuWxOHr7RAU3hL9WrGPmDmACLcBGAs/s0/p.png" alt="enter image description here" title="p.png"></p>



<h3 id="29-the-whole-cnn">2.9 The whole CNN</h3>

<p><img src="https://lh3.googleusercontent.com/-OuDdpGbIEDc/WivfRPXR6yI/AAAAAAAAAFQ/LZH707Lw7RksqdOIR49EsLBLj9QwSh_DACLcBGAs/s0/q.png" alt="enter image description here" title="q.png"></p>

<p><img src="https://lh3.googleusercontent.com/-7WUYZyLYfXs/WivjAytq6DI/AAAAAAAAAG8/affWGlCh_HEl687Vv7Pv9ltPdAoismfFgCLcBGAs/s0/v.png" alt="enter image description here" title="v.png"></p>



<h2 id="3-why-cnn">3. Why CNN</h2>

<ul>
<li><p>Some patterns are much smaller than the whole image.  <br>
<img src="https://lh3.googleusercontent.com/-9FLQCqfyOWw/WivgK7WwTyI/AAAAAAAAAFk/KG60O6Gpkug_Kf4xfTKj0WKrvssdnq-ygCLcBGAs/s0/r.png" alt="enter image description here" title="r.png"></p></li>
<li><p>The same patterns appear in different regions. <br>
<img src="https://lh3.googleusercontent.com/-3fabpqD56a8/WivgYAKQ6OI/AAAAAAAAAFs/VeU-hNdETlEE9k5kBdwmem6Amwpzgx6yACLcBGAs/s0/s.png" alt="enter image description here" title="s.png"></p></li>
<li><p>Subsampling the pixels will not change the object <br>
<img src="https://lh3.googleusercontent.com/-ZijUIGdmZpU/WivgtVbBp0I/AAAAAAAAAGI/ujDZAv_WFTMGRnZq0d3RfDS-heND6LR9ACLcBGAs/s0/t.png" alt="enter image description here" title="t.png"></p></li>
</ul>



<h2 id="soga">Soga~</h2>

<p><img src="https://lh3.googleusercontent.com/-XI7wZtxeE4w/Wivg9clXzmI/AAAAAAAAAGQ/eREvQ78goWcyW808lc4PTEYCC9-vVN82ACLcBGAs/s0/u.png" alt="enter image description here" title="u.png"></p>



<h2 id="4-对cnn的其他一些理解">4. 对CNN的其他一些理解</h2>



<h3 id="41-关于接受域receptive-field">4.1 关于接受域（receptive field）</h3>

<p>称在底层中影响上层输出单元 <script type="math/tex" id="MathJax-Element-5">s</script> 的单元集合为 <script type="math/tex" id="MathJax-Element-6">s</script> 的接受域(receptive field). <br>
<img src="https://lh3.googleusercontent.com/-0qaKzwQMhG4/Wivkh2ngmbI/AAAAAAAAAHU/7JshtCSdplELaS6jhMLUlSwIqxJjcpRYACLcBGAs/s0/w.png" alt="enter image description here" title="w.png"></p>

<blockquote>
  <p>处于卷积网络更深的层中的单元,它们的接受域要比处在浅层的单元的接受域更大。如果网络还包含类似步幅卷积或者池化之类的结构特征, 这种效应会加强。这意味着在卷积网络中尽管直接连接都是很稀疏的,但处在更深的层中的单元可以间接地连接到全部或者大部分输入图像。(表现性能)</p>
</blockquote>

<p><img src="https://lh3.googleusercontent.com/-T73eltpbGdw/WivkluABl6I/AAAAAAAAAHc/3YDAQHlk730xZ5Wf5BhxoT_1EIs3eDZfgCLcBGAs/s0/x.png" alt="enter image description here" title="x.png"></p>



<h3 id="42-卷积与池化作为一种无限强的先验">4.2 卷积与池化作为一种无限强的先验</h3>

<p>首先，弱先验具有较高的熵值，因此自由性较强．强先验具有较低的熵值，这样的先验在决定参数最终取值时可以起着非常积极的作用．</p>

<p>把卷积网络类比成全连接网络，但对于网络的权重具有无限强的先验．a) 所有隐藏单元的权重是共享的．b) 除了一些连续的小单元的权重外，其他的权重都是0．c) 池化也是一个无限强的先验：每个单元都具有对少量平移的不变性．</p>

<p><strong>卷积和池化可能导致欠拟合！</strong>　任何其他先验类似,卷积和池化只有当先验的假设合理且正确时才有用。如果一项任务依赖于保存精确的空间信息,那么在所有的特征上使用池化将会增大训练误差。</p>

<p><strong>根据实际需求选取先验</strong></p>



<h2 id="cnn实现">CNN实现</h2>



<h3 id="1-反向传播">1. 反向传播</h3>

<p>基本与FFNNs相同．</p>



<h3 id="2-共享权值的梯度问题">2. 共享权值的梯度问题</h3>

<p>一个常见的做法：取梯度的平均值</p>



<h3 id="3-cnn-in-keras">3. CNN in Keras</h3>

<p><img src="https://lh3.googleusercontent.com/-NsqxmGLDxRo/WivqzH18xTI/AAAAAAAAAIQ/i1NSOeWdOCky9UcVj9IMMBo7mXvV2zsowCLcBGAs/s0/y.png" alt="enter image description here" title="y.png"></p>

<p><img src="https://lh3.googleusercontent.com/-bA24w3VUuZc/WivrI0rqKLI/AAAAAAAAAIY/59Xd9-yWMUEHINaf3SblbAsesjzwRd5qgCLcBGAs/s0/z.png" alt="enter image description here" title="z.png"></p>