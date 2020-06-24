<TeXmacs|1.99.10>

<style|generic>

<\body>
  <section|Adjoint Method>

  Let <math|M> a manifold, and <math|x<around*|(|t|)>\<in\>C<rsup|1><around*|(|\<bbb-R\>,M|)>>
  a trajectory, obeying

  <\equation*>
    <frac|\<mathd\>x|\<mathd\>t><around*|(|t|)>=f<around*|(|t,x<around*|(|t|)>;\<theta\>|)>,
  </equation*>

  and

  <\equation*>
    x<around*|(|t<rsub|0>|)>=x<rsub|0>,
  </equation*>

  where <math|f\<in\>C<around*|(|\<bbb-R\>\<times\>M,T<rsub|M>|)>>
  parameterized by <math|\<theta\>>. For <math|\<forall\>t<rsub|1>\<gtr\>t<rsub|0>>,
  let

  <\equation*>
    x<rsub|1>\<assign\>x<rsub|0>+<big|int><rsub|t<rsub|0>><rsup|t<rsub|1>>f<around*|(|t,x<around*|(|t|)>;\<theta\>|)>\<mathd\>t.
  </equation*>

  Then

  <\theorem>
    <label|adjoint method> Let <math|<with|math-font|cal|C>\<in\>C<rsup|1><around*|(|M,\<bbb-R\>|)>>,
    and <math|\<forall\>x<around*|(|t|)>\<in\>C<rsup|1><around*|(|\<bbb-R\>,M|)>>
    obeying dynamics <math|f<around*|(|t,x;\<theta\>|)>\<in\>C<rsup|1><around*|(|\<bbb-R\>\<times\>M,T<rsub|M>|)>>
    with initial value <math|x<around*|(|t<rsub|0>|)>=x<rsub|0>>. Denote

    <\equation*>
      L\<assign\><with|math-font|cal|C><around*|(|x<rsub|0>+<big|int><rsub|t<rsub|0>><rsup|t<rsub|1>>f<around*|(|\<tau\>,x<around*|(|\<tau\>|)>;\<theta\>|)>\<mathd\>\<tau\>|)>.
    </equation*>

    Then we have, for <math|\<forall\>t\<in\><around*|[|t<rsub|0>,t<rsub|1>|]>>
    given,

    <\equation*>
      <frac|\<partial\>L|\<partial\>x<around*|(|t|)>>=<frac|\<partial\>L|\<partial\>x<rsub|1>>-<big|int><rsub|t><rsup|t<rsub|1>><frac|\<partial\>L|\<partial\>x<rsup|\<beta\>><around*|(|\<tau\>|)>><frac|\<partial\>f<rsup|\<beta\>>|\<partial\>x<rsup|\<alpha\>><around*|(|\<tau\>|)>><around*|(|\<tau\>,x<around*|(|\<tau\>|)>;\<theta\>|)>\<mathd\>\<tau\>,
    </equation*>

    and

    <\equation*>
      <frac|\<partial\>L|\<partial\>\<theta\>>=-<big|int><rsub|t<rsub|0>><rsup|t<rsub|1>><frac|\<partial\>L|\<partial\>x<rsup|\<beta\>><around*|(|\<tau\>|)>><frac|\<partial\>f<rsup|\<beta\>>|\<partial\>\<theta\>><around*|(|\<tau\>,x<around*|(|\<tau\>|)>;\<theta\>|)>\<mathd\>\<tau\>.
    </equation*>
  </theorem>

  <\proof>
    Suppose the <math|x<around*|(|t|)>> is layerized, the <math|L> depends on
    the variables (inputs and model parameters) on the <math|i>th layer can
    be regarded as the loss of a new model by truncating the original at the
    <math|i>th layer, which we call <math|L<rsub|i><around*|(|z<rsub|i>|)>>.

    <\align>
      <tformat|<table|<row|<cell|<frac|\<partial\>L<rsub|i>|\<partial\>x<rsup|\<alpha\>><rsub|i>><around*|(|x<rsub|i>|)>>|<cell|=<frac|\<partial\>L<rsub|i+1>|\<partial\>x<rsup|\<beta\>><rsub|i+1>><around*|(|x<rsub|i+1>|)><frac|\<partial\>x<rsup|\<beta\>><rsub|i+1>|\<partial\>x<rsup|\<alpha\>><rsub|i>><around*|(|x<rsub|i>|)>>>|<row|<cell|>|<cell|=<frac|\<partial\>L<rsub|i+1>|\<partial\>x<rsup|\<beta\>><rsub|1>><around*|(|x<rsub|i+1>|)><frac|\<partial\>|\<partial\>x<rsup|\<alpha\>><rsub|i>><around*|(|x<rsub|i><rsup|\<beta\>>+f<rsup|\<beta\>><around*|(|t<rsub|i>,x<rsub|i>;\<theta\>|)>\<Delta\>t|)>>>|<row|<cell|>|<cell|=<frac|\<partial\>L<rsub|i+1>|\<partial\>x<rsup|\<alpha\>><rsub|i+1>><around*|(|x<rsub|i+1>|)>+<frac|\<partial\>L<rsub|i+1>|\<partial\>x<rsup|\<beta\>><rsub|i+1>><around*|(|x<rsub|i+1>|)>\<partial\><rsub|\<alpha\>>f<rsup|\<beta\>><around*|(|t<rsub|i>,x<rsub|i>;\<theta\>|)>\<Delta\>t.>>>>
    </align>

    This hints that

    <\equation*>
      <frac|\<mathd\>|\<mathd\>t><frac|\<partial\>L|\<partial\>x<rsup|\<alpha\>><around*|(|t|)>>=-<frac|\<partial\>L|\<partial\>x<rsup|\<beta\>><around*|(|t|)>><frac|\<partial\>f<rsup|\<beta\>>|\<partial\>x<rsup|\<alpha\>><around*|(|t|)>><around*|(|t,x<around*|(|t|)>;\<theta\>|)>.
    </equation*>

    The initial value is <math|\<partial\>L/\<partial\>x<rsub|1>>. Thus

    <\equation*>
      <frac|\<partial\>L|\<partial\>x<around*|(|t|)>>=<frac|\<partial\>L|\<partial\>x<rsub|1>>-<big|int><rsub|t><rsup|t<rsub|1>><frac|\<partial\>L|\<partial\>x<rsup|\<beta\>><around*|(|\<tau\>|)>><frac|\<partial\>f<rsup|\<beta\>>|\<partial\>x<rsup|\<alpha\>><around*|(|\<tau\>|)>><around*|(|\<tau\>,x<around*|(|\<tau\>|)>;\<theta\>|)>\<mathd\>\<tau\>.
    </equation*>

    \;

    Varing <math|\<theta\>> will vary the
    <math|L<rsub|i><around*|(|x<rsub|i>|)>> from two aspects, the effect from
    <math|\<partial\>L<rsub|i+1>/\<partial\>\<theta\>> and the
    <math|\<Delta\>x<rsub|i+1>> caused by <math|\<Delta\>\<theta\>>.

    <\align>
      <tformat|<table|<row|<cell|<frac|\<partial\>L<rsub|i>|\<partial\>\<theta\>><around*|(|x<rsub|i>|)>>|<cell|=<frac|\<partial\>L<rsub|i+1>|\<partial\>\<theta\>><around*|(|x<rsub|i+1>|)>+<frac|\<partial\>L<rsub|i+1>|\<partial\>x<rsub|i+1>><frac|\<partial\>x<rsub|i+1>|\<partial\>\<theta\>>>>|<row|<cell|>|<cell|=<frac|\<partial\>L<rsub|i+1>|\<partial\>\<theta\>><around*|(|x<rsub|i+1>|)>+<frac|\<partial\>L<rsub|i+1>|\<partial\>x<rsub|i+1>><frac|\<partial\>|\<partial\>\<theta\>><around*|(|x<rsub|i><rsup|\<beta\>>+f<rsup|\<beta\>><around*|(|t<rsub|i>,x<rsub|i>;\<theta\>|)>\<Delta\>t|)>>>|<row|<cell|>|<cell|=<frac|\<partial\>L<rsub|i+1>|\<partial\>\<theta\>><around*|(|x<rsub|i+1>|)>+<frac|\<partial\>L<rsub|i+1>|\<partial\>x<rsub|i+1>><frac|\<partial\>f<rsup|\<beta\>>|\<partial\>\<theta\>><around*|(|t<rsub|i>,x<rsub|i>;\<theta\>|)>\<Delta\>t.>>>>
    </align>

    This hints that

    <\equation*>
      <frac|\<mathd\>|\<mathd\>t><frac|\<partial\>L|\<partial\>\<theta\>>=-<frac|\<partial\>L|\<partial\>x<rsup|\<alpha\>><around*|(|t|)>><frac|\<partial\>f<rsup|\<beta\>>|\<partial\>\<theta\>><around*|(|t,x<around*|(|t|)>,\<theta\>|)>.
    </equation*>

    The initial value is <math|0> since <math|<with|math-font|cal|C><around*|(|.|)>>
    is explicitly independent on <math|\<theta\>>. Thus

    <\equation*>
      <frac|\<partial\>L|\<partial\>\<theta\>>=-<big|int><rsub|t<rsub|0>><rsup|t><frac|\<partial\>L|\<partial\>x<rsup|\<beta\>><around*|(|\<tau\>|)>><frac|\<partial\>f<rsup|\<beta\>>|\<partial\>\<theta\>><around*|(|\<tau\>,x<around*|(|\<tau\>|)>;\<theta\>|)>\<mathd\>\<tau\>.
    </equation*>
  </proof>

  <section|Continuous-time Hopfield Network>

  <subsection|Discrete-time Hopfield Network>

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

  <subsection|Continuum of Time>

  Let's consider applying the convergence of Hopfield network to neural ODE
  for generic network architecture. This makes the discrete time <math|t> a
  continuum.

  <\theorem>
    <label|hopfield dynamics>Let <math|M> be a Riemann manifold. Given
    <math|<with|math-font|cal|E>\<in\>C<rsup|1><around*|(|M,\<bbb-R\>|)>>.
    For <math|\<forall\>x<around*|(|t|)>\<in\>C<rsup|1><around*|(|\<bbb-R\>,M|)>>
    s.t.

    <\equation*>
      <frac|d x<rsup|\<alpha\>>|d t><around*|(|t|)>=-\<nabla\><rsup|\<alpha\>><with|math-font|cal|E><around*|(|x<around*|(|t|)>|)>,
    </equation*>

    then <math|d<with|math-font|cal|E>/dt\<leqslant\>0> along
    <math|x<around*|(|t|)>>. Further, if <math|<with|math-font|cal|E>> is
    lower bounded, then <math|\<exists\>t<rsub|\<star\>>\<less\>+\<infty\>>,
    s.t. <math|\<mathd\>x<rsup|\<alpha\>>/\<mathd\>t=0> at
    <math|t<rsub|\<star\>>>.
  </theorem>

  <\proof>
    We have

    <\equation*>
      <frac|d<with|math-font|cal|E>|d t><around*|(|t|)>=\<nabla\><rsub|\<alpha\>><with|math-font|cal|E><around*|(|x<around*|(|t|)>|)><frac|d
      x<rsup|\<alpha\>>|d t><around*|(|t|)>=-\<nabla\><rsub|\<alpha\>><with|math-font|cal|E><around*|(|x<around*|(|t|)>|)>
      \<nabla\><rsup|\<alpha\>><with|math-font|cal|E><around*|(|x<around*|(|t|)>|)>\<leqslant\>0.
    </equation*>
  </proof>

  <\remark>
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
    W<rsub|\<alpha\> \ \<beta\>> x<rsup|\<beta\>>>, the analogy is obvious.
    The only differences are that the condition <math|W<rsub|\<alpha\>
    \<alpha\>>=0> and the <math|sign>-function are absent here.
  </remark>

  <subsection|Energy as Neural Network>

  The funciton <math|<with|math-font|cal|E>> is the energy in the Ising model
  (as a toy Hopfield network).

  <\theorem>
    Let <math|f<rsub|\<theta\>>> a neural network mapping from <math|M> to
    <math|\<bbb-R\>>, parameterized by <math|\<theta\>>, and
    <math|<with|math-font|cal|B>:M\<rightarrow\>D> where
    <math|D\<subseteq\>M> being compact. Then

    <\equation*>
      <with|math-font|cal|E><rsub|\<theta\>>\<assign\>f<rsub|\<theta\>>\<circ\><with|math-font|cal|B>
    </equation*>

    is a bounded function in <math|C<rsup|1><around*|(|M,\<bbb-R\>|)>>.
  </theorem>

  One option of <math|G> is <math|tanh>-function. However, the
  <math|tanh<around*|(|x|)>> will be saterated as
  <math|x\<rightarrow\>\<pm\>\<infty\>>. A better option is <slanted|boundary
  reflection>. Define boundary reflection map

  <\align>
    <tformat|<table|<row|<cell|f<rsub|BR>>|<cell|:\<bbb-R\><rsup|d>\<rightarrow\><around*|[|0,1|]><rsup|d>>>|<row|<cell|f<rsub|BR><around*|(|x|)>>|<cell|=<choice|<tformat|<table|<row|<cell|x,x\<in\><around*|[|0,1|]>>>|<row|<cell|-x,x\<in\><around*|[|-1,0|]>>>|<row|<cell|f<rsub|BR><around*|(|x-2|)>,x\<gtr\>1>>|<row|<cell|f<rsub|BR><around*|(|x+2|)>,x\<less\>-1>>>>>.>>>>
  </align>

  This function has constant gradient <math|\<pm\>1>, thus no saturation. It
  has periodic symmetry.
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|adjoint method|<tuple|1|1>>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|2>>
    <associate|auto-3|<tuple|2.1|2>>
    <associate|auto-4|<tuple|2.2|2>>
    <associate|auto-5|<tuple|2.3|3>>
    <associate|hopfield dynamics|<tuple|3|3>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Adjoint
      Method> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Continuum
      of Hopfield> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>Hopfield Network
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|2.2<space|2spc>Continuum
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|2.3<space|2spc>Energy as Neural Network
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>
    </associate>
  </collection>
</auxiliary>