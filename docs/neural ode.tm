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
  </definition>

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
    Let <math|<around*|(|x,W,b|)>> a discrete-time Hopfield network. Then
    <math|\<exists\>t<rsub|\<star\>>\<less\>+\<infty\>>, s.t.
    <math|x<around*|(|t+1|)>=x<around*|(|t|)>>.
  </theorem>

  <\proof>
    Since the states of the network are finite, the
    <math|<with|math-font|cal|E>> is lower bounded. Thus
    <math|\<exists\>t<rsub|\<star\>>\<less\>+\<infty\>>, s.t.
    <math|x<around*|(|t+1|)>=x<around*|(|t|)>>.
  </proof>

  <subsection|Continuous-time Hopfield Network>

  <\definition>
    [Continuous-time Hopfield Network]

    Let <math|t\<in\>\<bbb-N\>> and <math|x\<in\><around*|[|-1,+1|]><rsup|d>>,
    <math|W\<in\>\<bbb-R\><rsup|d>\<times\>\<bbb-R\><rsup|d>> with
    <math|W<rsub|\<alpha\> \<beta\>>=W<rsub|\<beta\> \<alpha\>>>, and
    <math|b\<in\>\<bbb-R\><rsup|d>>. Define dynamics

    <\equation*>
      \<tau\><frac|\<mathd\>x<rsup|\<alpha\>>|\<mathd\>t><around*|(|t|)>=-x<rsup|\<alpha\>><around*|(|t|)>+f<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
      x<rsup|\<beta\>><around*|(|t|)>+b<rsup|\<alpha\>>|)>,
    </equation*>

    where <math|\<tau\>> a constant and <math|f:\<bbb-R\>\<rightarrow\><around*|[|-1,1|]>>
    being increasing. The <math|<around*|(|x,W,b;\<tau\>,f|)>> is called a
    continuous-time Hopfield network.
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
    discrete-time.<\footnote>
      With experiments, we find that adding condition
      <math|W<rsub|\<alpha\>\<alpha\>>=0> for <math|\<forall\>\<alpha\>>
      significantly restricts the capacity of Hopfield network for learning,
      as well as its robustness.
    </footnote>
  </remark>

  <\theorem>
    Let <math|<around*|(|x,W,b;\<tau\>,f|)>> a continous-time Hopfield
    network. Then for <math|\<forall\>\<epsilon\>\<gtr\>0>,
    <math|\<exists\>t<rsub|\<star\>>\<less\>+\<infty\>>, s.t.
    <math|<around*|\<\|\|\>|\<mathd\>x/\<mathd\>t|\<\|\|\>>\<less\>\<epsilon\>>.
  </theorem>

  <\proof>
    The function <math|E\<assign\><with|math-font|cal|E>\<circ\>y> is lower
    bounded since <math|y>, i.e. function
    <math|f:\<bbb-R\>\<rightarrow\><around*|[|-1,1|]>>, is bounded. This
    <math|E> is a Lyapunov function for the continous-time Hopfield network.
  </proof>

  <\corollary>
    Let <math|<around*|(|x,W,b;\<tau\>,f|)>> a continous-time Hopfield
    network. And <math|D\<assign\><around*|{|x<rsub|n>\|x<rsub|n>\<in\>\<bbb-R\><rsup|d>,n=1,\<ldots\>,N|}>>
    a dataset<\footnote>
      We use Greek alphabet for component in <math|\<bbb-R\><rsup|d>> and
      Lattin alphabet for element in dataset.
    </footnote>. We can train the Hopfield nework by seeking a proper
    parameters <math|<around*|(|W,b|)>>, s.t. its stable point covers the
    dataset as much as possible, by

    <\algorithm>
      Given <math|1\<gtr\>\<Delta\>t\<gtr\>0>, and regularizer <math|R>,

      for step = 0,<text-dots>,S:

      <space|1em>for <math|x<rsub|n>\<in\>D>:

      <space|2em><math|y<around*|(|W,b|)>\<assign\>x<around*|(|t<rsub|0>+\<Delta\>t;W,b|)>>
      by solving the ODE of Hopfield network with IV
      <math|x<around*|(|t<rsub|0>|)>\<assign\>x<rsub|n>>

      <space|2em><math|loss<around*|(|W,b|)>\<assign\><around*|\<\|\|\>|y<around*|(|W,b|)>-x<rsub|n>|\<\|\|\>>+R<around*|(|W,b|)>>

      <space|2em>update <math|<around*|(|W,b|)>> by minimizing <math|loss>
      via gradient descent method.
    </algorithm>
  </corollary>

  <\proof>
    The model learns nothing with this algorithm if and only if the dynamics
    becomes identity transform. That is, for an arbitrary sample
    <math|x\<in\><around*|{|-1,1|}><rsup|d>>, when
    <math|x<rsup|\<alpha\>>=1>, <math|f<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
    x<rsup|\<beta\>>+b<rsup|\<alpha\>>|)>=1>; and when
    <math|x<rsup|\<alpha\>>=-1>, <math|f<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
    x<rsup|\<beta\>>+b<rsup|\<alpha\>>|)>=-1>. This can only be held when
    <math|<around*|\||W<rsub|\<alpha\>\<alpha\>>|\|>\<gg\><around*|\||W<rsub|\<alpha\>\<beta\>>|\|>>
    and <math|<around*|\||W<rsub|\<alpha\>\<alpha\>>|\|>\<gg\>b<rsub|\<alpha\>>>
    for <math|\<forall\>\<alpha\>> and <math|\<forall\>\<beta\>\<neq\>\<alpha\>>.
    Indeed,

    <\equation*>
      x<rsup|\<alpha\>>=1\<Rightarrow\>f<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<beta\>>
      x<rsup|\<beta\>>+b<rsup|\<alpha\>>|)>\<approx\>f<around*|(|W<rsup|\<alpha\>><rsub|<space|2.4spc>\<alpha\>>
      x<rsup|\<alpha\>>|)>\<approx\>1;
    </equation*>

    and the same holds for <math|x<rsup|\<alpha\>>=-1>. With a proper
    weight-initializer and regularizer, this will never happen. So, with this
    algorithm, Hopfield network can memorize the samples, s.t. its stable
    point covers the dataset as much as possible.
  </proof>
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|adjoint method|<tuple|1|1>>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|1.1|2>>
    <associate|auto-3|<tuple|2|2>>
    <associate|auto-4|<tuple|2.1|2>>
    <associate|auto-5|<tuple|2.2|3>>
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
    </associate>
  </collection>
</auxiliary>