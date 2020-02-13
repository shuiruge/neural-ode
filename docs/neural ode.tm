<TeXmacs|1.99.11>

<style|generic>

<\body>
  <math|L\<assign\>L<around*|(|z<around*|(|t|)>,t,\<theta\>|)>>

  <math|<wide|z|\<dot\>><rsup|\<alpha\>><around*|(|t|)>=:f<rsup|\<alpha\>><around*|(|z<around*|(|t|)>,t,\<theta\>|)>>

  <math|\<Rightarrow\>z<rsup|\<alpha\>><around*|(|t+\<epsilon\>|)>=z<rsup|\<alpha\>><around*|(|t|)>+\<epsilon\>f<rsup|\<alpha\>><around*|(|z<around*|(|t|)>,t,\<theta\>|)>>

  <math|\<Rightarrow\><frac|\<partial\>z<rsup|\<beta\>><around*|(|t+\<epsilon\>|)>|\<partial\>z<rsup|\<alpha\>><around*|(|t|)>>=\<delta\><rsup|\<beta\>><rsub|\<alpha\>>+\<epsilon\><frac|\<partial\>f<rsup|\<beta\>>|\<partial\>z<rsup|\<alpha\>><around*|(|t|)>><around*|(|z<around*|(|t|)>,t,\<theta\>|)>>

  <math|a<rsub|\<alpha\>><around*|(|t|)>\<assign\><frac|\<partial\>L|\<partial\>z<rsup|\<alpha\>><around*|(|t|)>>>

  <math|<frac|\<partial\>L|\<partial\>z<rsup|\<alpha\>><around*|(|t|)>>=<frac|\<partial\>L|\<partial\>z<rsup|\<beta\>><around*|(|t+\<epsilon\>|)>><frac|\<partial\>z<rsup|\<beta\>><around*|(|t+\<epsilon\>|)>|\<partial\>z<rsup|\<alpha\>><around*|(|t|)>>>

  <math|\<Rightarrow\>a<rsub|\<alpha\>><around*|(|t|)>=a<rsub|\<beta\>><around*|(|t+\<epsilon\>|)><frac|\<partial\>z<rsup|\<beta\>><around*|(|t+\<epsilon\>|)>|\<partial\>z<rsup|\<alpha\>><around*|(|t|)>>=<around*|[|a<rsub|\<beta\>><around*|(|t|)>+\<epsilon\><wide|a|\<dot\>><rsub|\<beta\>><around*|(|t|)>|]><around*|[|\<delta\><rsup|\<beta\>><rsub|\<alpha\>>+\<epsilon\><frac|\<partial\>f<rsup|\<beta\>>|\<partial\>z<rsup|\<alpha\>><around*|(|t|)>><around*|(|z<around*|(|t|)>,t,\<theta\>|)>|]>>

  <math|\<Rightarrow\><wide|a|\<dot\>><rsub|\<alpha\>><around*|(|t|)>=-a<rsub|\<beta\>><around*|(|t|)><frac|\<partial\>f<rsup|\<beta\>>|\<partial\>z<rsup|\<alpha\>><around*|(|t|)>><around*|(|z<around*|(|t|)>,t,\<theta\>|)>>

  <subsection|Calculation of <math|\<partial\>L/\<partial\>\<theta\>>>

  Suppose the model is layerized, the loss depends on the variables (inputs
  and model parameters) on the <math|i>th layer can be regarded as the loss
  of a <with|color|blue|<with|color|red|new>> model by truncating the
  original at the <math|i>th layer, which we call
  <math|L<rsub|i><around*|(|z<rsub|i>|)>>. Varing <math|\<theta\>> will vary
  the <math|L<rsub|0><around*|(|z<rsub|0>|)>> from two aspects, the effect
  from <math|dL<rsub|1>/d\<theta\>> and the <math|\<Delta\>z<rsub|1>> caused
  by <math|\<Delta\>\<theta\>>.

  <math|<frac|dL<rsub|0>|d\<theta\>><around*|(|z<rsub|0>|)>=<frac|dL<rsub|1>|d\<theta\>><around*|(|z<rsub|1>|)>+<frac|dL<rsub|1>|dz<rsub|1>><frac|\<partial\>z<rsub|1>|\<partial\>\<theta\>>>.

  The same relation holds for any <math|i>, by simply considering a truncated
  model,

  <math|<frac|dL<rsub|i>|d\<theta\>><around*|(|z<rsub|0>|)>=<frac|dL<rsub|i+1>|d\<theta\>><around*|(|z<rsub|i+1>|)>+<frac|dL<rsub|i+1>|dz<rsub|i+1>><frac|\<partial\>z<rsub|i+1>|\<partial\>\<theta\>>>.

  Thus we have, recursely,

  <math|<tabular|<tformat|<table|<row|<cell|<frac|dL<rsub|0>|d\<theta\>><around*|(|z<rsub|0>|)>>|<cell|=>|<cell|<frac|dL<rsub|1>|d\<theta\>><around*|(|z<rsub|1>|)>+<frac|dL<rsub|1>|dz<rsub|1>><frac|\<partial\>z<rsub|1>|\<partial\>\<theta\>>>>|<row|<cell|>|<cell|=>|<cell|<around*|[|<frac|dL<rsub|2>|d\<theta\>><around*|(|z<rsub|1>|)>+<frac|dL<rsub|2>|dz<rsub|2>><frac|\<partial\>z<rsub|2>|\<partial\>\<theta\>>|]>+<frac|dL<rsub|1>|dz<rsub|1>><frac|\<partial\>z<rsub|1>|\<partial\>\<theta\>>>>|<row|<cell|>|<cell|=>|<cell|<frac|dL<rsub|2>|d\<theta\>><around*|(|z<rsub|1>|)>+<frac|dL<rsub|2>|dz<rsub|2>><frac|\<partial\>z<rsub|2>|\<partial\>\<theta\>>+<frac|dL<rsub|1>|dz<rsub|1>><frac|\<partial\>z<rsub|1>|\<partial\>\<theta\>>>>|<row|<cell|>|<cell|=>|<cell|\<ldots\>.>>|<row|<cell|>|<cell|=>|<cell|<frac|dL<rsub|N>|d\<theta\>><around*|(|z<rsub|1>|)>+<big|sum><rsub|i=1><rsup|N><frac|dL<rsub|i>|dz<rsub|i>><frac|\<partial\>z<rsub|i>|\<partial\>\<theta\>>.>>>>>>

  By <math|z<rsub|i+1><around*|(|t|)>=z<rsub|i><around*|(|t|)>+\<epsilon\>f<around*|(|z<rsub|i><around*|(|t|)>,t;\<theta\>|)>>,
  <math|\<partial\>z<rsub|i>/\<partial\>\<theta\>=\<epsilon\>\<partial\>f<around*|(|z<rsub|i>,t;\<theta\>|)>/\<partial\>\<theta\>>

  <section|Continuum of Hopfield>

  <subsection|Hopfield Network>

  Consider Hopfield network. Let <math|x<around*|(|t|)>\<in\><around*|{|-1,+1|}><rsup|N>>
  denotes the state of the network at descrete time <math|t=0,1,\<ldots\>>;
  and <math|W> a matrix on <math|\<bbb-R\><rsup|N>>, essentially ensuring
  <math|W<rsub|\<alpha\> \<beta\>>=W<rsub|\<beta\> \<alpha\>>> and
  <math|W<rsub|\<alpha\> \<alpha\>>=0>. Define energy
  <math|E<rsub|W><around*|(|x<around*|(|t|)>|)>\<assign\>-W<rsub|\<alpha\>\<beta\>>
  x<rsup|\<alpha\>><around*|(|t|)> x<rsup|\<beta\>><around*|(|t|)>>.

  <\theorem>
    Along dynamics <math|x<rsub|\<alpha\>><around*|(|t+1|)>=sign<around*|[|W<rsub|\<alpha\>
    \<beta\>> x<rsup|\<beta\>><around*|(|t|)>|]>>,
    <math|E<rsub|W><around*|(|x<around*|(|t+1|)>|)>-E<rsub|W><around*|(|x<around*|(|t|)>|)>\<leqslant\>0>.
  </theorem>

  <\proof>
    Consider the async-updation of Hopfield network. Let's change the
    component at dimension <math|<wide|\<alpha\>|^>>, i.e.
    <math|x<rprime|'><rsub|<wide|\<alpha\>|^>>=sign<around*|[|W<rsub|<wide|\<alpha\>|^>
    \<beta\>> x<rsup|\<beta\>>|]>>, then

    <\align>
      <tformat|<table|<row|<cell|E<rsub|W><around*|(|x<rprime|'>|)>-E<rsub|W><around*|(|x|)>=>|<cell|-W<rsub|\<alpha\>\<beta\>>
      x<rprime|'><rsup|\<alpha\>>x<rprime|'><rsup|\<beta\>>+W<rsub|\<alpha\>\<beta\>>
      x<rsup|\<alpha\>>x<rsup|\<beta\>>>>|<row|<cell|=>|<cell|-2
      <around*|(|x<rprime|'><rsup|<wide|\<alpha\>|^>>-x<rsup|<wide|\<alpha\>|^>>|)>
      W<rsub|<wide|\<alpha\>|^> \<beta\>> x<rsup|\<beta\>>,>>>>
    </align>

    which employs conditions <math|W<rsub|\<alpha\> \<beta\>>=W<rsub|\<beta\>
    \<alpha\>>> and <math|W<rsub|\<alpha\> \<alpha\>>=0>. Next, we prove
    that, combining with <math|x<rprime|'><rsub|<wide|\<alpha\>|^>>=sign<around*|[|W<rsub|<wide|\<alpha\>|^>
    \<beta\>> x<rsup|\<beta\>>|]>>, this implies
    <math|E<rsub|W><around*|(|x<rprime|'>|)>-E<rsub|W><around*|(|x|)>\<leqslant\>0>.

    If <math|<around*|(|x<rprime|'><rsup|<wide|\<alpha\>|^>>-x<rsup|<wide|\<alpha\>|^>>|)>\<gtr\>0>,
    then <math|x<rprime|'><rsup|<wide|\<alpha\>|^>>=1> and
    <math|x<rsup|<wide|\<alpha\>|^>>=-1>. Since
    <math|x<rprime|'><rsub|<wide|\<alpha\>|^>>=sign<around*|[|W<rsub|<wide|\<alpha\>|^>
    \<beta\>> x<rsup|\<beta\>>|]>>, <math|W<rsub|<wide|\<alpha\>|^> \<beta\>>
    x<rsup|\<beta\>>\<gtr\>0>. Then <math|E<rsub|W><around*|(|x<rprime|'>|)>-E<rsub|W><around*|(|x|)>\<less\>0>.
    Contrarily, if <math|<around*|(|x<rprime|'><rsup|<wide|\<alpha\>|^>>-x<rsup|<wide|\<alpha\>|^>>|)>\<less\>0>,
    then <math|x<rprime|'><rsup|<wide|\<alpha\>|^>>=-1> and
    <math|x<rsup|<wide|\<alpha\>|^>>=1>, implying
    <math|W<rsub|<wide|\<alpha\>|^> \<beta\>> x<rsup|\<beta\>>\<less\>0>.
    Also <math|E<rsub|W><around*|(|x<rprime|'>|)>-E<rsub|W><around*|(|x|)>\<less\>0>.
    Otherwise, <math|E<rsub|W><around*|(|x<rprime|'>|)>-E<rsub|W><around*|(|x|)>=0>.
    So, we conclude <math|E<rsub|W><around*|(|x<rprime|'>|)>-E<rsub|W><around*|(|x|)>\<leqslant\>0>.
  </proof>

  Since the states of the network are finite, the <math|E<rsub|W>> is lower
  bounded. Thus the network converges (relaxes) at finite <math|t>.

  <subsection|Continuum>

  Let's consider applying the convergence of Hopfield network to neural ODE
  for generic network architecture. This makes the descrete time <math|t> a
  continuum.

  <\theorem>
    Let <math|M> be a Riemann manifold with metric <math|g>. Given any
    function <math|<with|math-font|cal|E>\<in\>C<rsup|1><around*|(|M,\<bbb-R\>|)>>.
    Let <math|x<around*|(|t|)>\<in\>C<rsup|1><around*|(|\<bbb-R\>,M|)>>
    denote trajectory. Then, <math|d<with|math-font|cal|E>/dt\<leqslant\>0>
    along <math|x<around*|(|t|)>> if

    <\equation*>
      <frac|d x<rsup|\<alpha\>>|d t><around*|(|t|)>=-\<nabla\><rsup|\<alpha\>><with|math-font|cal|E><around*|(|x<around*|(|t|)>|)>.
    </equation*>
  </theorem>

  <\proof>
    We have

    <\equation*>
      <frac|d<with|math-font|cal|E>|d t><around*|(|t|)>=\<nabla\><rsub|\<alpha\>><with|math-font|cal|E><around*|(|x<around*|(|t|)>|)><frac|d
      x<rsup|\<alpha\>>|d t><around*|(|t|)>=-\<nabla\><rsub|\<alpha\>><with|math-font|cal|E><around*|(|x<around*|(|t|)>|)>
      \<nabla\><rsup|\<alpha\>><with|math-font|cal|E><around*|(|x<around*|(|t|)>|)>\<leqslant\>0.
    </equation*>
  </proof>

  Further, if the function <math|<with|math-font|cal|E>> is lower bounded,
  then the trajectory converges (relaxes) at finite <math|t>. We call this
  dynamic with lower bounded <math|<with|math-font|cal|E>> as \PHopfield
  dynamic with energy <math|<with|math-font|cal|E>>\Q.

  This is the continuum analogy to the convergence of Hopfield network.
  Indeed, let <math|M> be <math|\<bbb-R\><rsup|N>>, and
  <math|<with|math-font|cal|E><around*|(|x|)>=-W<rsub|\<alpha\> \<beta\>>
  x<rsup|\<alpha\>> x<rsup|\<beta\>>> with <math|W<rsub|\<alpha\>
  \<beta\>>=W<rsub|\<beta\> \<alpha\>>>, then dynamics becomes

  <\equation*>
    <frac|d x<rsub|\<alpha\>>|d t><around*|(|t|)>=-\<nabla\><rsub|\<alpha\>><with|math-font|cal|E><around*|(|x<around*|(|t|)>|)>=-2
    W<rsub|\<alpha\> \<beta\>> x<rsup|\<beta\>><around*|(|t|)>,
  </equation*>

  which makes

  <\equation*>
    <frac|d<with|math-font|cal|E>|d t><around*|(|t|)>=-2<frac|d
    x<rsup|\<alpha\>>|d t><around*|(|t|)> W<rsub|\<alpha\> \<beta\>>
    x<rsup|\<beta\>><around*|(|t|)>.
  </equation*>

  Comparing with the proof of convergence of Hopfield network, i.e.
  <math|\<Delta\>E<rsub|W><around*|(|x|)>=-2 \<Delta\>x<rsup|\<alpha\>>
  W<rsub|\<alpha\> \ \<beta\>> x<rsup|\<beta\>>>, the analogy is obvious. The
  only differences are that the condition <math|W<rsub|\<alpha\>
  \<alpha\>>=0> and the <math|sign>-function are absent here.

  It is known that a Riemann manifold can be locally re-coordinated to be
  Euclidean. Thus, locally <math|\<exists\><wide|x|^>> coordinate, s.t.

  <\equation*>
    <frac|d x<rsup|\<alpha\>>|d t><around*|(|t|)>=-\<delta\><rsup|><rsup|\<alpha\>\<beta\>><frac|\<partial\><with|math-font|cal|E>|\<partial\>x<rsup|\<beta\>>><around*|(|x<around*|(|t|)>|)>.
  </equation*>
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|1|?>>
    <associate|auto-3|<tuple|1.1|?>>
    <associate|auto-4|<tuple|1.2|?>>
    <associate|auto-5|<tuple|1.3|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <with|par-left|<quote|1tab>|1<space|2spc>Calculation of
      <with|mode|<quote|math>|\<partial\>L/\<partial\>\<theta\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Continuum
      of Hopfield> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Hopfield Network
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|1.2<space|2spc>Continuum
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|1.3<space|2spc>Quadratic Form and
      Homogeneousness <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>
    </associate>
  </collection>
</auxiliary>