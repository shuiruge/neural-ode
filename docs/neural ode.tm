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
    <label|hopfield dynamics>Let <math|M> be a Riemann manifold with metric
    <math|g>. Given any function <math|<with|math-font|cal|E>\<in\>C<rsup|1><around*|(|M,\<bbb-R\>|)>>.
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

  <subsection|General Form>

  In the proof of theorem <reference|hopfield dynamics>,

  <\equation*>
    <frac|d<with|math-font|cal|E>|d t><around*|(|t|)>=\<nabla\><rsub|\<alpha\>><with|math-font|cal|E><around*|(|x<around*|(|t|)>|)><frac|d
    x<rsup|\<alpha\>>|d t><around*|(|t|)>.
  </equation*>

  We try to find the generic form of <math|d x<rsup|\<alpha\>>/d t> that
  ensures <math|d <with|math-font|cal|E>/d t\<leqslant\>0>. To restrict the
  formation, symmetries are called for. Denote

  <\equation*>
    <frac|d x<rsup|\<alpha\>>|d t>=F<rsup|\<alpha\>><around*|[|<with|math-font|cal|E>|]><around*|(|x|)>,
  </equation*>

  where operator <math|F:C<rsup|\<infty\>><around*|(|M,M|)>\<mapsto\>C<around*|(|M,M|)>>.

  <\axiom>
    Locality.
  </axiom>

  This implies:

  <\equation*>
    F<around*|[|<with|math-font|cal|E>|]>=F<around*|(|<with|math-font|cal|E>,\<nabla\><with|math-font|cal|E>,\<nabla\><rsup|2><with|math-font|cal|E>,\<ldots\>.|)>;
  </equation*>

  <\axiom>
    <math|<with|math-font|cal|E>\<rightarrow\><with|math-font|cal|E>+C> for
    any constant <math|C>.
  </axiom>

  Combining with the previous, this then implies

  <\equation*>
    F<around*|[|<with|math-font|cal|E>|]>=F<around*|(|\<nabla\><with|math-font|cal|E>,\<nabla\><rsup|2><with|math-font|cal|E>,\<ldots\>.|)>.
  </equation*>

  <\axiom>
    Co-variance.
  </axiom>

  This implies the balance of index. Thus

  <\equation*>
    F<rsup|\<alpha\>><around*|[|<with|math-font|cal|E>|]>=c<rsub|1>\<nabla\><rsup|\<alpha\>><with|math-font|cal|E>+c<rsub|3>\<nabla\><rsup|\<alpha\>><with|math-font|cal|E><around*|(|\<nabla\><rsup|\<beta\>><with|math-font|cal|E>\<nabla\><rsub|\<beta\>><with|math-font|cal|E>|)>+c<rsub|3><rprime|'>\<nabla\><rsup|\<alpha\>><with|math-font|cal|E><around*|(|\<nabla\><rsup|\<beta\>>\<nabla\><rsub|\<beta\>><with|math-font|cal|E>|)>+<with|math-font|cal|O><around*|(|\<nabla\><rsup|5>|)>.
  </equation*>

  <\axiom>
    <label|axiom of numerical stability>For <math|x\<rightarrow\>\<lambda\>
    x>, <math|\<exists\>k\<less\>m\<less\>M\<less\>K> s.t.
    <math|m\<less\>F<around*|[|<with|math-font|cal|E>|]><around*|(|x|)>\<less\>M>
    for any <math|\<lambda\>>, where <math|k> and <math|K> are numerically
    finite. E.g. <math|k\<sim\>1> and <math|K\<sim\>10>. This is essential
    for numerical stability, i.e. no under- and over-flow.
  </axiom>

  First, we have to notice a property of the feed forward neural network with
  rectified activations (e.g. ReLU, leaky ReLU, and linear).

  <\lemma>
    Rectified activations are linearly homogeneous.
  </lemma>

  <\lemma>
    If <math|f> and g are homogeneous with order <math|\<lambda\><rsub|f>>
    and <math|\<lambda\><rsub|g>> respectively, then <math|f\<circ\>g> is
    homogeneous with order <math|\<lambda\><rsub|f>+\<lambda\><rsub|g>>.\ 
  </lemma>

  <\theorem>
    <label|homogenity>Let <math|f<rsub|nn><around*|(|x;\<theta\>|)>> a feed
    forward nerual network with rectified activations, where <math|\<theta\>>
    represents the parameters (weights and biases). At the initial stage of
    training, <math|f<rsub|nn><around*|(|.;\<theta\>|)>> is linearly
    homogeneous.<strong|> That is

    <\equation*>
      f<rsub|nn><around*|(|\<lambda\>x;\<theta\><rsub|ini>|)>=\<lambda\>f<around*|(|x;\<theta\><rsub|ini>|)>.
    </equation*>
  </theorem>

  <\proof>
    Notice that <math|f<rsub|nn><around*|(|.;\<theta\>|)>> is linearly
    homogeneous when its biases vanish, and that biases are initialized as
    zeros. So <math|f<rsub|nn><around*|(|.;\<theta\>|)>> is linearly
    homogeneous at initial stage of training.
  </proof>

  If <math|<with|math-font|cal|E>> is constructed by such neural network,
  <math|F<around*|[|<with|math-font|cal|E>|]>> can be further simplified.
  Indeed, if <math|<with|math-font|cal|E><around*|(|x;\<theta\>|)>\<assign\><sqrt|f<rsub|\<alpha\>><around*|(|x;\<theta\>|)>
  f<rsup|\<alpha\>><around*|(|x;\<theta\>|)>>>, then
  <math|<with|math-font|cal|E><around*|(|\<lambda\>
  x;\<theta\><rsub|ini>|)>=\<lambda\> <with|math-font|cal|E><around*|(|x;\<theta\><rsub|ini>|)>>,
  implying <math|F<rsup|\<alpha\>><around*|[|<with|math-font|cal|E>|]>=c<rsub|1>\<nabla\><rsup|\<alpha\>><with|math-font|cal|E>+c<rsub|3>\<nabla\><rsup|\<alpha\>><with|math-font|cal|E><around*|(|\<nabla\><rsup|\<beta\>><with|math-font|cal|E>\<nabla\><rsub|\<beta\>><with|math-font|cal|E>|)>+<with|math-font|cal|O><around*|(|\<nabla\><rsup|5>|)>>,
  which scales as <math|\<lambda\><rsup|0>>.<\footnote>
    Numerical experiment on MNIST dataset shows that this configuration
    indeed out-performs than others, like
    <math|<with|math-font|cal|E><around*|(|x;\<theta\>|)>\<assign\>f<rsub|\<alpha\>><around*|(|x;\<theta\>|)>
    f<rsup|\<alpha\>><around*|(|x;\<theta\>|)>>,
    <math|<with|math-font|cal|E><around*|(|x;\<theta\>|)>\<assign\>f<rsup|2><around*|(|x;\<theta\>|)>>,
    and non-Hopfield, e.t.c. In this experiment, <math|c<rsub|1>=5> and
    <math|c<rsub|i\<gtr\>1>\<equiv\>0>; Nadam optimizer is employed, with
    standard parameters, except for <math|\<epsilon\>=10<rsup|-3>>; the
    dimension of <math|x> is <math|64>. For the details, c.f. the file
    <samp|node/experiments/Hopfield.ipynb>.
  </footnote>

  Alternatively, if <math|<with|math-font|cal|E><around*|(|x;\<theta\>|)>\<assign\>f<rsup|2><around*|(|x;\<theta\>|)>>,
  then <math|<with|math-font|cal|E><around*|(|\<lambda\>
  x;\<theta\><rsub|ini>|)>=\<lambda\><rsup|2>
  <with|math-font|cal|E><around*|(|x;\<theta\><rsub|ini>|)>>. In this case,
  axiom <reference|axiom of numerical stability> can never be satisfied.
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|adjoint method|<tuple|1|?>>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|2|?>>
    <associate|auto-3|<tuple|2.1|?>>
    <associate|auto-4|<tuple|2.2|?>>
    <associate|auto-5|<tuple|2.3|?>>
    <associate|auto-6|<tuple|2.2|?>>
    <associate|auto-7|<tuple|2.3|?>>
    <associate|axiom of numerical stability|<tuple|7|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnr-1|<tuple|1|?>>
    <associate|homogenity|<tuple|10|?>>
    <associate|hopfield dynamics|<tuple|3|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Adjoint
      Method> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Calculation of
      <with|mode|<quote|math>|\<partial\>L/\<partial\>x>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|1.2<space|2spc>Calculation of
      <with|mode|<quote|math>|\<partial\>L/\<partial\>\<theta\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Continuum
      of Hopfield> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>Hopfield Network
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1tab>|2.2<space|2spc>Continuum
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|2.3<space|2spc>General Form
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>
    </associate>
  </collection>
</auxiliary>