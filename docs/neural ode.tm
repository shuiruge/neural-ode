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

  <section|Energy Based>

  <\lemma>
    Given an arbitrary lower bounded function
    <math|<with|math-font|cal|E>\<in\>C<rsup|1><around*|(|\<bbb-R\><rsup|N>,\<bbb-R\>|)>>,
    for any trajectory <math|z<around*|(|t|)>\<in\>C<rsup|1><around*|(|\<bbb-R\>,\<bbb-R\><rsup|N>|)>>,
    <math|d<with|math-font|cal|E>/dt\<leqslant\>0> along
    <math|z<around*|(|t|)>> if

    <\equation*>
      <frac|dz<rsup|\<alpha\>>|dt><around*|(|t|)>=-<frac|\<partial\><with|math-font|cal|E>|\<partial\>z<rsup|\<alpha\>>><around*|(|z<around*|(|t|)>|)>.
    </equation*>
  </lemma>

  <\proof>
    <\equation*>
      <frac|d<with|math-font|cal|E>|dt>=<big|sum><rsub|\<alpha\>><frac|\<partial\><with|math-font|cal|E>|\<partial\>z<rsup|\<alpha\>>><around*|(|z<around*|(|t|)>|)><frac|dz<rsup|\<alpha\>>|dt><around*|(|t|)>=-<around*|\<\|\|\>|<frac|\<partial\><with|math-font|cal|E>|\<partial\>z<rsup|\<alpha\>>><around*|(|z<around*|(|t|)>|)>|\<\|\|\>><rsub|2><rsup|2>\<leqslant\>0.
    </equation*>
  </proof>

  <\definition>
    Given an arbitrary positive defined linear transformation <math|W> on
    <math|\<bbb-R\><rsup|N>>, and an arbitrary function
    <math|f\<in\>C<rsup|1><around*|(|\<bbb-R\><rsup|N>,\<bbb-R\><rsup|N>|)>>,
    for <math|\<forall\>z\<in\>\<bbb-R\><rsup|N>>, define energy

    <\equation*>
      E<rsub|<around*|(|W,f|)>><around*|(|z|)>\<assign\><big|sum><rsub|\<alpha\>,\<beta\>>W<rsub|\<alpha\>\<beta\>>
      f<rsup|\<alpha\>><around*|(|z|)> f<rsup|\<beta\>><around*|(|z|)>.
    </equation*>
  </definition>

  <\lemma>
    Energy <math|E<rsub|<around*|(|W,f|)>>> is lower bounded.
  </lemma>

  <\theorem>
    Given the energy <math|E<rsub|<around*|(|W,f|)>>> on
    <math|\<bbb-R\><rsup|N>>, for any trajectory
    z<around*|(|t|)>\<in\>C<rsup|1><around*|(|\<bbb-R\>,\<bbb-R\><rsup|N>|)>,
    <math|E<rsub|<around*|(|W,f|)>><around*|(|z<around*|(|t|)>|)>> decreases
    monotonically, converging to its lower bound, iff

    <\equation*>
      <frac|dz<rsup|\<gamma\>>|dt><around*|(|t|)>=-<big|sum><rsub|\<alpha\>,\<beta\>>W<rsub|\<alpha\>\<beta\>>
      f<rsup|\<alpha\>><around*|(|z<around*|(|t|)>|)>
      <frac|\<partial\>f<rsup|\<beta\>>|\<partial\>z<rsup|\<gamma\>>><around*|(|z<around*|(|t|)>|)>.
    </equation*>
  </theorem>

  \;

  \;
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|1|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <with|par-left|<quote|1tab>|1<space|2spc>Calculation of
      <with|mode|<quote|math>|\<partial\>L/\<partial\>\<theta\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>
    </associate>
  </collection>
</auxiliary>