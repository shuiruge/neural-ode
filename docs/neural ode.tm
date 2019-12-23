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
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
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