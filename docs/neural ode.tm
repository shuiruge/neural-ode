<TeXmacs|1.99.10>

<style|generic>

<\body>
  <section|Neural ODE>

  <subsection|Adjoint Method>

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
      <frac|\<partial\>L|\<partial\>x<rsup|\<alpha\>><around*|(|t|)>>=<frac|\<partial\>L|\<partial\>x<rsup|\<alpha\>><rsub|1>>-<big|int><rsub|t><rsup|t<rsub|1>><frac|\<partial\>L|\<partial\>x<rsup|\<beta\>><around*|(|\<tau\>|)>><frac|\<partial\>f<rsup|\<beta\>>|\<partial\>x<rsup|\<alpha\>>><around*|(|\<tau\>,x<around*|(|\<tau\>|)>;\<theta\>|)>\<mathd\>\<tau\>,
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

  <section|Hopfield Network>

  <subsection|Discrete-time Hopfield Network>

  <subsubsection|Definition>

  <\definition>
    [Discrete-time Hopfield Network]

    Let <math|t\<in\>\<bbb-N\>> and <math|x\<in\><around*|{|-1,+1|}><rsup|d>>,
    <math|W\<in\>\<bbb-R\><rsup|d>\<times\>\<bbb-R\><rsup|d>> with
    <math|W<rsub|\<alpha\> \<beta\>>=W<rsub|\<beta\> \<alpha\>>> and
    <math|W<rsub|\<alpha\> \<alpha\>>=0>, and
    <math|b\<in\>\<bbb-R\><rsup|d>>. Define discrete-time dynamics

    <\equation*>
      x<rsup|\<alpha\>><around*|(|t+1|)>=sign<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
      x<rsup|\<beta\>><around*|(|t|)>+b<rsup|\<alpha\>>|)>.
    </equation*>

    The <math|<around*|(|x,W,b|)>> is called a discrete-time Hopfield
    network.
  </definition>

  <subsubsection|Convergence>

  <\lemma>
    Let <math|<around*|(|x,W,b|)>> a discrete-time Hopfield network. Define
    <math|<with|math-font|cal|E><around*|(|x|)>\<assign\>-<around*|(|1/2|)>W<rsub|\<alpha\>\<beta\>>
    x<rsup|\<alpha\>> x<rsup|\<beta\>>-b<rsub|\<alpha\>>x<rsup|\<alpha\>>>.
    Then <math|<with|math-font|cal|E><around*|(|x<around*|(|t+1|)>|)>-<with|math-font|cal|E><around*|(|x<around*|(|t|)>|)>\<leqslant\>0>.
  </lemma>

  <\proof>
    Consider async-updation of Hopfield network, that is, change the
    component at dimension <math|<wide|\<alpha\>|^>>, i.e.
    <math|x<rprime|'><rsub|<wide|\<alpha\>|^>>=sign<around*|[|W<rsub|<wide|\<alpha\>|^>
    \<beta\>> x<rsup|\<beta\>>+b<rsub|<wide|\<alpha\>|^>>|]>>, then

    <\align>
      <tformat|<table|<row|<cell|<with|math-font|cal|E><around*|(|x<rprime|'>|)>-<with|math-font|cal|E><around*|(|x|)>=>|<cell|-<frac|1|2>W<rsub|\<alpha\>\<beta\>>
      x<rprime|'><rsup|\<alpha\>>x<rprime|'><rsup|\<beta\>>-b<rsub|\<alpha\>>x<rprime|'><rsup|\<alpha\>>+<frac|1|2>W<rsub|\<alpha\>\<beta\>>
      x<rsup|\<alpha\>>x<rsup|\<beta\>>+b<rsub|\<alpha\>>x<rsup|\<alpha\>>>>|<row|<cell|=>|<cell|-2
      <around*|(|x<rprime|'><rsup|<wide|\<alpha\>|^>>-x<rsup|<wide|\<alpha\>|^>>|)>
      <around*|(|W<rsub|<wide|\<alpha\>|^> \<beta\>>
      x<rsup|\<beta\>>+b<rsub|<wide|\<alpha\>|^>>|)>,>>>>
    </align>

    which employs conditions <math|W<rsub|\<alpha\> \<beta\>>=W<rsub|\<beta\>
    \<alpha\>>> and <math|W<rsub|\<alpha\> \<alpha\>>=0>. Next, we prove
    that, combining with <math|x<rprime|'><rsub|<wide|\<alpha\>|^>>=sign<around*|[|W<rsub|<wide|\<alpha\>|^>
    \<beta\>> x<rsup|\<beta\>>+b<rsub|<wide|\<alpha\>|^>>|]>>, this implies
    <math|<with|math-font|cal|E><around*|(|x<rprime|'>|)>-<with|math-font|cal|E><around*|(|x|)>\<leqslant\>0>.

    If <math|<around*|(|x<rprime|'><rsup|<wide|\<alpha\>|^>>-x<rsup|<wide|\<alpha\>|^>>|)>\<gtr\>0>,
    then <math|x<rprime|'><rsup|<wide|\<alpha\>|^>>=1> and
    <math|x<rsup|<wide|\<alpha\>|^>>=-1>. Since
    <math|x<rprime|'><rsub|<wide|\<alpha\>|^>>=sign<around*|[|W<rsub|<wide|\<alpha\>|^>
    \<beta\>> x<rsup|\<beta\>>+b<rsub|<wide|\<alpha\>|^>>|]>>,
    <math|W<rsub|<wide|\<alpha\>|^> \<beta\>>
    x<rsup|\<beta\>>+b<rsub|<wide|\<alpha\>|^>>\<gtr\>0>. Then
    <math|<with|math-font|cal|E><around*|(|x<rprime|'>|)>-<with|math-font|cal|E><around*|(|x|)>\<less\>0>.
    Contrarily, if <math|<around*|(|x<rprime|'><rsup|<wide|\<alpha\>|^>>-x<rsup|<wide|\<alpha\>|^>>|)>\<less\>0>,
    then <math|x<rprime|'><rsup|<wide|\<alpha\>|^>>=-1> and
    <math|x<rsup|<wide|\<alpha\>|^>>=1>, implying
    <math|W<rsub|<wide|\<alpha\>|^> \<beta\>>
    x<rsup|\<beta\>>+b<rsub|<wide|\<alpha\>|^>>\<less\>0>. Also
    <math|<with|math-font|cal|E><around*|(|x<rprime|'>|)>-<with|math-font|cal|E><around*|(|x|)>\<less\>0>.
    Otherwise, <math|<with|math-font|cal|E><around*|(|x<rprime|'>|)>-<with|math-font|cal|E><around*|(|x|)>=0>.
    So, we conclude <math|<with|math-font|cal|E><around*|(|x<rprime|'>|)>-<with|math-font|cal|E><around*|(|x|)>\<leqslant\>0>.
  </proof>

  <\theorem>
    [Convergene of Discrete-time Hopfield Network] Let
    <math|<around*|(|x,W,b|)>> a discrete-time Hopfield network. Then any
    trajectory obeying the update rule will converge either to a fixed point
    or a limit circle.
  </theorem>

  <\proof>
    Since the states of the network are finite, the
    <math|<with|math-font|cal|E>> is lower bounded.
  </proof>

  <subsection|Continuous-time Hopfield Network>

  <subsubsection|Definition>

  <\definition>
    [Continuous-time Hopfield Network]

    Let <math|t\<in\><around*|[|0,+\<infty\>|)>> and
    <math|x\<in\><around*|[|-1,+1|]><rsup|d>>,
    <math|W\<in\>\<bbb-R\><rsup|d>\<times\>\<bbb-R\><rsup|d>> with
    <math|W<rsub|\<alpha\> \<beta\>>=W<rsub|\<beta\> \<alpha\>>>, and
    <math|b\<in\>\<bbb-R\><rsup|d>>. Define dynamics

    <\equation*>
      \<tau\><frac|\<mathd\>x<rsup|\<alpha\>>|\<mathd\>t><around*|(|t|)>=-x<rsup|\<alpha\>><around*|(|t|)>+f<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
      x<rsup|\<beta\>><around*|(|t|)>+b<rsup|\<alpha\>>|)>,
    </equation*>

    where <math|\<tau\>\<in\><around*|(|0,+\<infty\>|)>> a constant and
    <math|f:\<bbb-R\>\<rightarrow\><around*|[|-1,1|]>> being increasing. The
    <math|<around*|(|x,W,b;\<tau\>,f|)>> is called a continuous-time Hopfield
    network.
  </definition>

  <\remark>
    With

    <\equation*>
      \<tau\><frac|x<rsup|\<alpha\>><around*|(|t+\<Delta\>t|)>-x<rsup|\<alpha\>><around*|(|t|)>|\<Delta\>t>=<around*|\<nobracket\>|-x<rsup|\<alpha\>><around*|(|t|)>+f<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
      x<rsup|\<beta\>><around*|(|t|)>+b<rsup|\<alpha\>>|)>|)>.
    </equation*>

    Setting <math|\<Delta\>t=\<tau\>> gives and
    <math|f<around*|(|.|)>=sign<around*|(|.|)>> gives

    <\equation*>
      x<rsup|\<alpha\>><around*|(|t+\<tau\>|)>=sign<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
      x<rsup|\<beta\>><around*|(|t|)>+b<rsup|\<alpha\>>|)>,
    </equation*>

    which is the same as the discrete-time Hopfield network.
  </remark>

  <subsubsection|Convergence>

  <\lemma>
    Let <math|<around*|(|x,W,b;\<tau\>,f|)>> a continous-time Hopfield
    network. Define <math|a<rsup|\<alpha\>>\<assign\>W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
    x<rsup|\<beta\>>+b<rsup|\<alpha\>>> and
    <math|y<rsup|\<alpha\>>\<assign\>f<around*|(|a<rsup|\<alpha\>>|)>>, then

    <\equation*>
      <with|math-font|cal|E><around*|(|y|)>\<assign\>-<frac|1|2>W<rsub|\<alpha\>\<beta\>>y<rsup|\<alpha\>>y<rsup|\<beta\>>-b<rsub|\<alpha\>>y<rsup|\<alpha\>>+<big|sum><rsub|\<alpha\>><big|int><rsup|y<rsup|\<alpha\>>>f<rsup|-1><around*|(|y<rsup|\<alpha\>>|)>\<mathd\>y<rsup|\<alpha\>>.
    </equation*>

    Then <math|<with|math-font|cal|E><around*|(|y<around*|(|x<around*|(|t+\<mathd\>t|)>|)>|)>-<with|math-font|cal|E><around*|(|y<around*|(|x<around*|(|t|)>|)>|)>\<leqslant\>0>.
  </lemma>

  <\proof>
    The dynamics of <math|a<rsup|\<alpha\>>> is

    <\align>
      <tformat|<table|<row|<cell|\<tau\><frac|\<mathd\>a<rsup|\<alpha\>>|\<mathd\>t>>|<cell|=\<tau\>W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
      <frac|\<mathd\>x<rsup|\<beta\>>|\<mathd\>t>>>|<row|<cell|>|<cell|=W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>><around*|[|-x<rsup|\<beta\>><around*|(|t|)>+f<around*|(|a<rsup|\<beta\>>|)>|]>>>|<row|<cell|>|<cell|=-<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>x<rsup|\<beta\>><around*|(|t|)>+b<rsup|\<alpha\>>|)>+b<rsup|\<alpha\>>+W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>y<rsup|\<beta\>>>>|<row|<cell|>|<cell|=W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>y<rsup|\<beta\>>+b<rsup|\<alpha\>>-a<rsup|\<alpha\>>.>>>>
    </align>

    Since <math|W> is symmetric, we have <math|\<partial\><with|math-font|cal|E>/\<partial\>y<rsup|\<alpha\>>=-W<rsub|\<alpha\>\<beta\>>y<rsup|\<beta\>>-b<rsub|\<alpha\>>+f<rsup|-1><around*|(|y<rsub|\<alpha\>>|)>>.
    Then

    <\align>
      <tformat|<table|<row|<cell|<frac|\<mathd\><with|math-font|cal|E>|\<mathd\>t>>|<cell|=<frac|\<mathd\>y<rsup|\<alpha\>>|\<mathd\>t><around*|(|-W<rsub|\<alpha\>\<beta\>>y<rsup|\<beta\>>-b<rsub|\<alpha\>>+f<rsup|-1><around*|(|y<rsub|\<alpha\>>|)>|)>>>|<row|<cell|>|<cell|=<frac|\<mathd\>y<rsup|\<alpha\>>|\<mathd\>t><around*|(|-W<rsub|\<alpha\>\<beta\>>y<rsup|\<beta\>>-b<rsub|\<alpha\>>+a<rsub|\<alpha\>>|)>>>|<row|<cell|>|<cell|=-<frac|\<mathd\>y<rsup|\<alpha\>>|\<mathd\>t><around*|(|W<rsub|\<alpha\>\<beta\>>y<rsup|\<beta\>>+b<rsub|\<alpha\>>-a<rsub|\<alpha\>>|)>>>>>
    </align>

    Notice that, the second term of rhs is exactly the dynamics of
    <math|a<rsub|\<alpha\>>>, then

    <\align>
      <tformat|<table|<row|<cell|<frac|\<mathd\><with|math-font|cal|E>|\<mathd\>t>>|<cell|=-\<tau\><frac|\<mathd\>y<rsup|\<alpha\>>|\<mathd\>t><frac|\<mathd\>a<rsub|\<alpha\>>|\<mathd\>t>>>|<row|<cell|>|<cell|=-\<tau\><frac|\<mathd\>y<rsup|\<alpha\>>|\<mathd\>a<rsup|\<alpha\>>><around*|(|<frac|\<mathd\>a<rsup|\<alpha\>>|\<mathd\>t><frac|\<mathd\>a<rsub|\<alpha\>>|\<mathd\>t>|)>>>|<row|<cell|>|<cell|=-\<tau\>f<rprime|'><around*|(|a<rsup|\<alpha\>>|)><around*|(|<frac|\<mathd\>a<rsup|\<alpha\>>|\<mathd\>t><frac|\<mathd\>a<rsub|\<alpha\>>|\<mathd\>t>|)>.>>>>
    </align>

    Since <math|f> is increasing and <math|\<tau\>\<gtr\>0>,
    <math|\<mathd\><with|math-font|cal|E>/\<mathd\>t\<leqslant\>0>.
  </proof>

  <\remark>
    The condition <math|W<rsub|\<alpha\>\<alpha\>>=0> for
    <math|\<forall\>\<alpha\>> is not essential for this lemma. Indeed, this
    condition is absent in the proof. This differs from the case of
    discrete-time.
  </remark>

  <\theorem>
    [Convergene of Continuous-time Hopfield Network] Let
    <math|<around*|(|x,W,b;\<tau\>,f|)>> a continous-time Hopfield network.
    Then any trajectory along the dynamics will converge either to a fixed
    point or a limit circle.
  </theorem>

  <\proof>
    The function <math|E\<assign\><with|math-font|cal|E>\<circ\>y> is lower
    bounded since <math|y>, i.e. function
    <math|f:\<bbb-R\>\<rightarrow\><around*|[|-1,1|]>>, is bounded. This
    <math|E> is a Lyapunov function for the continous-time Hopfield network.
  </proof>

  <subsubsection|Learning Rule>

  <\corollary>
    Let <math|<around*|(|x,W,b;\<tau\>,f|)>> a continous-time Hopfield
    network. And <math|D\<assign\><around*|{|x<rsub|n>\|x<rsub|n>\<in\>\<bbb-R\><rsup|d>,n=1,\<ldots\>,N|}>>
    a dataset<\footnote>
      We use Greek alphabet for component in <math|\<bbb-R\><rsup|d>> and
      Lattin alphabet for element in dataset.
    </footnote>. If add constraint <math|W<rsub|\<alpha\>\<alpha\>>=0> for
    <math|\<forall\>\<alpha\>>, then we can train the Hopfield nework by
    seeking a proper parameters <math|<around*|(|W,b|)>>, s.t. its stable
    points cover the dataset as much as possible, by<\footnote>
      This algorithm generalizes the algorithm 42.9 of Mackay.
    </footnote>

    <\algorithm>
      <\python-code>
        W, b = init_W, init_b \ # e.g. by Glorot initializer

        for step in range(max_step):

        \ \ \ \ for x in dataset:

        \ \ \ \ \ \ \ \ y = f(W @ x + b)

        \ \ \ \ \ \ \ \ loss = norm(x - y)

        \ \ \ \ \ \ \ \ optimizer.minimize(objective=loss, variables=(W, b))

        \ \ \ \ \ \ \ \ W = set_zero_diag(symmetrize(W))
      </python-code>
    </algorithm>
  </corollary>

  <\proof>
    For <math|\<forall\>x<rsub|n>\<in\>D>, we try to find
    <math|<around*|(|W,b|)>>, s.t. <math|\<mathd\>x/\<mathd\>t=0> at
    <math|x<rsub|n>>, i.e.

    <\equation*>
      x<rsub|n><rsup|\<alpha\>>=f<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
      x<rsub|n><rsup|\<beta\>>+b<rsup|\<alpha\>>|)>.
    </equation*>

    When <math|W<rsub|\<alpha\>\<alpha\>>=0> for <math|\<forall\>\<alpha\>>,
    <math|f<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
    x<rsup|\<beta\>>+b<rsup|\<alpha\>>|)>> thus has no information of
    <math|x<rsup|\<alpha\>>>, it has to predict the <math|x<rsup|\<alpha\>>>
    by the interaction between <math|x<rsup|\<alpha\>>> and the other
    <math|x>'s components.
  </proof>

  <\remark>
    This algorithm is equivalent to

    <\algorithm>
      <\python-code>
        dt = ... \ # e.g. 0.1

        W, b = init_W, init_b

        for step in range(max_step):

        \ \ \ \ for x in dataset:

        \ \ \ \ \ \ \ \ # that is, compute x(dt), with x(0) = x

        \ \ \ \ \ \ \ \ y = ode_solve(f=lambda t, x: -x + f(W @ x + b), t0=0,
        t1=dt, x0=x)

        \ \ \ \ \ \ \ \ loss = norm(x - y)

        \ \ \ \ \ \ \ \ optimizer.minimize(objective=loss, variables=(W, b))

        \ \ \ \ \ \ \ \ W = set_zero_diag(symmetrize(W))
      </python-code>
    </algorithm>

    Indeed, trying to reach <math|y=x> within a small interval will force
    <math|x> to be a fixed point.
  </remark>

  <subsubsection|Relation to Auto-encoder>

  Notice that at fixed point <math|x<rsub|\<star\>>>,
  <math|x<rsub|\<star\>><rsup|\<alpha\>>=f<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
  x<rsub|\<star\>><rsup|\<beta\>>+b<rsup|\<alpha\>>|)>>, which is a simple
  auto-encoder.

  <subsubsection|Fixed Points>

  We study the stability of fixed points. Let
  <math|z<rsup|\<alpha\>>\<assign\>W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
  x<rsup|\<beta\>>+b<rsup|\<alpha\>>>. Jacobian

  <\align>
    <tformat|<table|<row|<cell|J<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
    >|<cell|=<frac|\<partial\>|\<partial\>x<rsup|\<beta\>>><around*|(|-x<rsup|\<alpha\>>+f<around*|(|z<rsup|\<alpha\>>|)>|)>>>|<row|<cell|>|<cell|=-\<delta\><rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>+f<rprime|'><around*|(|z<rsup|\<alpha\>>|)>W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>.>>>>
  </align>

  If <math|f<around*|(|x|)>=tanh<around*|(|x|)>>, and at fixed point,

  <\align>
    <tformat|<table|<row|<cell|J<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
    >|<cell|=-\<delta\><rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>+<frac|1|2><around*|(|1-f<rsup|2><around*|(|z<rsup|\<alpha\>>|)>|)>W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>>>|<row|<cell|>|<cell|=-\<delta\><rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>+<frac|1|2><around*|(|1-x<rsub|\<star\>><rsup|\<alpha\>>|)><around*|(|1+x<rsub|\<star\>><rsup|\<alpha\>>|)>W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>.>>>>
  </align>

  The eigen-value of <math|J>, <math|\<lambda\><rsub|J>=:-1+\<lambda\>>, have

  <\equation*>
    det<around*|(|<frac|1|2><around*|(|1-x<rsub|\<star\>><rsup|\<alpha\>>|)><around*|(|1+x<rsub|\<star\>><rsup|\<alpha\>>|)>W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>-\<lambda\>\<delta\><rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>|)>=0
  </equation*>

  Because of the linearity of this equation, and
  <math|<around*|\||x<rsub|\<star\>><rsup|\<alpha\>>|\|>\<less\>1> being
  bounded for <math|\<forall\>\<alpha\>>, we can expect that
  <math|\<lambda\>\<sim\>\<lambda\><rsub|W>>, where <math|\<lambda\><rsub|W>>
  is the eigen-value of <math|W>. If <math|<around*|\||\<lambda\><rsub|W>|\|>\<ll\>1>,
  then <math|<around*|\||\<lambda\>|\|>\<ll\>1>, and then
  <math|\<lambda\><rsub|J>\<approx\>-1>, indicating that the fixed points are
  stable.

  <section|Variations>

  <subsection|Variation 1>

  <\theorem>
    Let <math|v\<in\>\<bbb-R\><rsup|d>>, <math|F\<in\>C<rsup|1><around*|(|\<bbb-R\><rsup|n>,\<bbb-R\>|)>>,
    <math|W\<in\>\<bbb-R\><rsup|n>\<times\>\<bbb-R\><rsup|d>>,
    <math|b\<in\>\<bbb-R\><rsup|n>>, and <math|\<tau\>\<gtr\>0>. Define the
    dynamics

    <\equation*>
      \<tau\><frac|\<mathd\>x|\<mathd\>t>=-\<nabla\>E<around*|(|x|)>=-x+W<rsup|T>\<cdot\>\<nabla\>F<around*|(|W\<cdot\>x+b|)>+v.
    </equation*>

    If <math|\<nabla\>F<around*|(|.|)>> is bounded, i.e.
    <math|\<exists\>K\<gtr\>0> s.t. <math|max<rsub|x\<in\>\<bbb-R\><rsup|n>><around*|{|\<nabla\>F<around*|(|x|)>|}>\<less\>K>,
    then any trajectory along the dynamics will converge either to a fixed
    point or a limit circle.
  </theorem>

  <\proof>
    Let <math|E<around*|(|x|)>\<assign\><frac|1|2>x<rsub|\<alpha\>>x<rsup|\<alpha\>>-v<rsub|\<alpha\>>x<rsup|\<alpha\>>-F<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
    x<rsup|\<beta\>>+b<rsup|\<alpha\>>|)>>, then
    <math|\<tau\>\<mathd\>x/\<mathd\>t=-\<nabla\>E<around*|(|x|)>>. The
    <math|-x> term will dominate the <math|W<rsup|T>\<cdot\>\<nabla\>F<around*|(|W\<cdot\>x+b|)>>
    term for <math|<around*|\<\|\|\>|x|\<\|\|\>>\<gtr\>K<around*|\<\|\|\>|W|\<\|\|\>>>,
    thus converges. So <math|E> is a Lyapunov function of the dynamics.
  </proof>

  <\example>
    Let <math|F<around*|(|x|)>\<assign\><big|sum><rsub|\<alpha\>><big|int><rsup|x<rsup|\<alpha\>>>\<sigma\><around*|(|s|)>\<mathd\>s>,
    where <math|\<sigma\>> is sigmoid function. Then

    <\equation*>
      \<tau\><frac|\<mathd\>x|\<mathd\>t>=-x+W<rsup|T>\<cdot\>\<sigma\><around*|(|W\<cdot\>x+b|)>+v.
    </equation*>

    This coincides with the form in ref <reference|On autoencoder scoring>.
  </example>

  <\example>
    Let <math|F<around*|(|x|)>\<assign\>\<beta\><rsup|-1>ln<around*|(|\<beta\><big|sum><rsub|\<alpha\>>\<mathe\><rsup|x<rsup|\<alpha\>>>|)>>,
    <math|b=0>, and <math|v=0>, then

    <\equation*>
      \<tau\><frac|\<mathd\>x|\<mathd\>t>=-x+W<rsup|T>\<cdot\>softmax<around*|(|\<beta\>W\<cdot\>x|)>.
    </equation*>

    This coincides with the form in ref <reference|Hopfield networks is All
    You Need>.
  </example>

  <\example>
    <label|example: softmax>Let <math|v<rsub|i>\<assign\>W<rsub|i,\<cdummy\>>>,
    i.e. the <math|i>th row of the matrix <math|W>. Assume
    <math|<around*|\<\|\|\>|v<rsub|i>|\<\|\|\>>=1> for
    <math|\<forall\>i=1,\<ldots\>,n>. Let
    <math|F<around*|(|x|)>\<assign\>\<beta\><rsup|-1>ln<around*|(|\<beta\><big|sum><rsub|\<alpha\>>\<mathe\><rsup|x<rsup|\<alpha\>>>|)>>,
    and <math|v=0>, then

    <\equation*>
      \<tau\><frac|\<mathd\>x<rsup|\<alpha\>>|\<mathd\>t>=-x<rsup|\<alpha\>>+<big|sum><rsub|i>p<rsub|i>v<rsup|\<alpha\>><rsub|i>,
    </equation*>

    where <math|z<rsub|i>\<assign\>v<rsub|i>\<cdot\>x+b<rsub|i>> and then
    <math|p<rsup|i>\<assign\>exp<around*|(|\<beta\>z<rsup|i>|)>/<big|sum><rsub|j>exp<around*|(|\<beta\>z<rsup|j>|)>>.
    The <math|<around*|{|<around*|(|p<rsub|i>,v<rsub|i>|)>\|i=1,\<ldots\>,n|}>>
    forms a categorical distribution.
  </example>

  <\lemma>
    Assume example <reference|example: softmax>. The Jacobian of the dynamics
    is

    <\equation*>
      J<rsup|\<alpha\>\<beta\>><around*|(|x|)>=-\<delta\><rsup|\<alpha\>\<beta\>>+Cov<rsub|p<around*|(|x|)>><around*|(|v<rsup|\<alpha\>>,v<rsup|\<beta\>>|)>,
    </equation*>

    where <math|Cov<rsub|p><around*|(|\<cdummy\>,\<cdummy\>|)>> denotes the
    covariance given distribution <math|p>.
  </lemma>

  <\proof>
    Directly,

    <\align>
      <tformat|<table|<row|<cell|J<rsup|\<alpha\>\<beta\>>>|<cell|\<equiv\><frac|\<partial\>|\<partial\>x<rsub|\<beta\>>><around*|(|-x<rsup|\<alpha\>>+<big|sum><rsub|i>v<rsup|\<alpha\>><rsub|i>p<rsub|i>|)>>>|<row|<cell|>|<cell|=-\<delta\><rsup|\<alpha\>\<beta\>>+<big|sum><rsub|i,j>v<rsup|\<alpha\>><rsub|i><frac|\<partial\>p<rsub|i>|\<partial\>z<rsup|j>><frac|\<partial\>z<rsup|j>|\<partial\>x<rsub|\<beta\>>>>>|<row|<cell|>|<cell|=-\<delta\><rsup|\<alpha\>\<beta\>>+\<beta\><big|sum><rsub|i,j>v<rsup|\<alpha\>><rsub|i>v<rsub|j><rsup|\<beta\>><around*|(|p<rsub|i>\<delta\><rsub|i,j>-p<rsub|i>p<rsub|j>|)>>>|<row|<cell|>|<cell|=-\<delta\><rsup|\<alpha\>\<beta\>>+\<beta\><big|sum><rsub|i>p<rsub|i>v<rsub|i><rsup|\<alpha\>>v<rsub|i><rsup|\<beta\>>-\<beta\><around*|(|<big|sum><rsub|i>p<rsub|i>v<rsub|i><rsup|\<alpha\>>|)><around*|(|<big|sum><rsub|j>p<rsub|j>v<rsub|j><rsup|\<beta\>>|)>>>|<row|<cell|>|<cell|=-\<delta\><rsup|\<alpha\>\<beta\>>+\<beta\>\<bbb-E\><around*|(|v<rsup|\<alpha\>>v<rsup|\<beta\>>|)>-\<beta\>\<bbb-E\><around*|(|v<rsup|\<alpha\>>|)>\<bbb-E\><around*|(|v<rsup|\<beta\>>|)>>>|<row|<cell|>|<cell|=-\<delta\><rsup|\<alpha\>\<beta\>>+\<beta\>Cov<rsub|p><around*|(|v<rsup|\<alpha\>>,v<rsup|\<beta\>>|)>.>>>>
    </align>

    And notice that the only variable that depends on <math|x> is <math|p>.
    So we insert <math|x> and gain the result.
  </proof>

  For instance, at fixed point <math|x=v<rsub|1>>,
  <math|p=<around*|(|1,0,\<ldots\>,0|)>>.
  <math|Cov<rsub|p<around*|(|v<rsub|1>|)>><around*|(|v<rsup|\<alpha\>>,v<rsup|\<beta\>>|)>=v<rsub|1><rsup|\<alpha\>>v<rsub|1><rsup|\<beta\>>-v<rsub|1><rsup|\<alpha\>>v<rsub|1><rsup|\<beta\>>=0>.
  So <math|J<rsup|\<alpha\>\<beta\>>=-\<delta\><rsup|\<alpha\>\<beta\>>> is
  negative defined, indicating that the fixed point is stable.

  <section|References>

  <\enumerate-numeric>
    <item><label|On autoencoder scoring> <hlink|On autoencoder
    scoring|http://proceedings.mlr.press/v28/kamyshanska13.pdf>.

    <item><label|Hopfield networks is All You Need> <hlink|Hopfield networks
    is All You Need|https://arxiv.org/abs/2008.02217>.
  </enumerate-numeric>
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|Hopfield networks is All You Need|<tuple|2|?>>
    <associate|On autoencoder scoring|<tuple|1|?>>
    <associate|adjoint method|<tuple|1|1>>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-10|<tuple|2.2.3|?>>
    <associate|auto-11|<tuple|2.2.4|?>>
    <associate|auto-12|<tuple|2.2.5|?>>
    <associate|auto-13|<tuple|3|?>>
    <associate|auto-14|<tuple|3.1|?>>
    <associate|auto-15|<tuple|4|?>>
    <associate|auto-2|<tuple|1.1|2>>
    <associate|auto-3|<tuple|2|2>>
    <associate|auto-4|<tuple|2.1|2>>
    <associate|auto-5|<tuple|2.1.1|3>>
    <associate|auto-6|<tuple|2.1.2|?>>
    <associate|auto-7|<tuple|2.2|?>>
    <associate|auto-8|<tuple|2.2.1|?>>
    <associate|auto-9|<tuple|2.2.2|?>>
    <associate|example: softmax|<tuple|15|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnote-2|<tuple|2|?>>
    <associate|footnr-1|<tuple|1|?>>
    <associate|footnr-2|<tuple|2|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Neural
      ODE> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Adjoint Method
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Hopfield
      Network> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>Discrete-time Hopfield
      Network <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|2.2<space|2spc>Continuous-time Hopfield
      Network <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Variations>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6><vspace|0.5fn>

      <with|par-left|<quote|1tab>|3.1<space|2spc>Variation 1
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>References>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>