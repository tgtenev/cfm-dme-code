#!/usr/bin/perl
# quality_split.pl: the scatter about the radial-acceleration relation in the SPARC galaxies of quality flag Q = 2
# against those of Q = 1, below and above the scale, for Paper II §6.5 (disturbed galaxies).
#
# Paper II §6.5: the median per-galaxy ratio of the scatter below a0 (y = g_bar/a0 < 1) to the scatter above it
# (y >= 1) is 0.648 for Q = 2 against 0.452 for Q = 1 (quality_split-results.txt line 11), a difference erased or
# reversed in 12 to 25 percent of the bootstrap resamplings: the fraction of draws in which the ratio of Q = 2 to
# Q = 1 is at or below 1 is 0.1232 for the ratio of the medians and 0.2517 for the ratio of the pooled scatters
# (lines 17 and 18). Q = 2 marks minor asymmetries and/or H I data of lower quality, Q = 1 high-quality data;
# Q = 3 is excluded, as in every fit of the deposit. Slow re-equilibration in disturbed disks would raise the
# ratio of Q = 2 above that of Q = 1, whereas poorer data raise the scatter of Q = 2 at every y alike, so the ratio
# within each class separates the two.
#
# Each galaxy is fitted at a fixed scale with MOND's simple function by the per-galaxy fit of two_constant_fit.pl:
# distance and inclination profiled under their catalog priors through q, the disk ratio and each bulge ratio free
# under lognormal priors of 0.10 dex about 0.5 and 0.7, and the converged optimizer. The fitting functions below are
# copied unchanged from two_constant_fit.pl, whose header describes them. The scale is the simple function's fit
# there, a0 = 1.0796e-10 m/s^2 (two_constant_fit-results.txt line 13). A point's residual is
# log10(g_obs/q^2) - log10(nu(y) g_bar), and a scatter is the rms of the residuals about their mean. The errors come
# from 4000 bootstrap draws of the galaxies within each class.
#
# Options (--key=value): --a0=value (default 1.0796e-10); --bulge=free|fixed (a free bulge ratio per galaxy, the
# default, or Y_b = 1.4 Y_d); --frozen=1 (distances and inclinations at catalog values); --priorb=dex (the bulge
# prior's width); --tol, --cap (the optimizer's tolerance and round cap).
# Inputs: data/sparc/SPARC_Lelli2016c.mrt and the mass models in data/sparc/rotmod_ltg/. Run from the repository
# root, in about 20 seconds; the defaults, --bulge=free --a0=1.0796e-10, are those of the results file:
#   perl rotation-curves/quality_split.pl data/sparc/SPARC_Lelli2016c.mrt data/sparc/rotmod_ltg \
#        > rotation-curves/quality_split-results.txt
use strict; use warnings;
$| = 1;

my $CONV=3.24078e-14; my $FLOOR=0.05; my $Y0=0.5; my $BRAT=1.4; our $PRIOR=0.10;
my $PI=4*atan2(1,1); my $D2R=$PI/180.0;
my $INV_LN10=1.0/log(10.0); sub log10 { log($_[0])*$INV_LN10 } my $LGY0=log10($Y0);
my $YB0=0.7; my $LGYB0=log10($YB0);          # Li et al. 2018 bulge prior centre

my %OPT=(bulge=>'free',frozen=>0,priorb=>'',tol=>1e-7,cap=>50,a0=>'1.0796e-10');
my @POS;
for my $a (@ARGV) { if ($a=~/^--([\w-]+)=(.*)$/) { (my $k=$1)=~tr/-/_/; die "unknown option --$1\n" unless exists $OPT{$k}; $OPT{$k}=$2 }
                    else { push @POS,$a } }
my ($t1path,$rotdir,@EXTRA)=@POS;
die "usage: perl rotation-curves/quality_split.pl SPARC_Lelli2016c.mrt rotmod_ltg [--key=value ...]\n"
    if !defined $rotdir || @EXTRA;
our $PRIORB = $OPT{priorb} ne '' ? $OPT{priorb} : $PRIOR;
our $FROZEN = $OPT{frozen} ? 1 : 0;
my $FREEB = $OPT{bulge} eq 'free' ? 1 : 0;
die "--bulge must be free or fixed\n" unless $OPT{bulge}=~/^(free|fixed)$/;
my $TOL=$OPT{tol}; my $CAP=$OPT{cap};
my $A0=$OPT{a0};

# ---- the per-galaxy fit of two_constant_fit.pl: the functions this script uses, copied unchanged ----
# --- Table 1: distance, inclination, their errors, quality ---
open my $t1,'<',$t1path or die "Table1: $!";
my (%DST,%EDST,%INC,%EINC,%Q); my $rule=0;
while (my $l=<$t1>) { $l=~s/\r?\n$//;
    if ($l=~/^-+$/){$rule++;next} next if $rule<4;
    my @f=split ' ',$l; next unless @f>=18 && $f[0]=~/^[A-Za-z0-9]/;
    next unless $f[5]=~/^[\d.]+$/ && $f[17]=~/^\d+$/;
    $DST{$f[0]}=$f[2]; $EDST{$f[0]}=$f[3]; $INC{$f[0]}=$f[5]; $EINC{$f[0]}=$f[6]; $Q{$f[0]}=$f[17];
}
close $t1;

# --- rotmod files -> per-galaxy point arrays + nuisance-prior constants ---
# per point: [gg, gs, Vo, invE2, sR, go, gd, gb]: gas, disk + 1.4 bulge (the fixed bulge), V_obs, weight,
#   sqrt(R) in the units of the accelerations, g_obs, and the disk and bulge separately (the free bulge)
opendir my $dh,$rotdir or die; my @files=sort grep{/_rotmod\.dat$/}readdir $dh; closedir $dh;
my (%GAL,%NUI,%BUL); my ($ngal,$npts)=(0,0);
for my $fn (@files) {
    (my $name=$fn)=~s/_rotmod\.dat$//;
    next unless exists $INC{$name}; next if $Q{$name}==3; next if $INC{$name}<30;
    open my $fh,'<',"$rotdir/$fn" or next; my @pts; my $hasb=0;
    while (my $l=<$fh>) { next if $l=~/^#/; $l=~s/\r?\n$//;
        my @c=split ' ',$l; next unless @c>=6 && $c[0]=~/^[\d.]/ && $c[0]>0;
        my ($R,$Vo,$eV,$Vg,$Vd,$Vb)=@c[0..5];
        $hasb=1 if $Vb!=0;
        push @pts,[$Vg*abs($Vg)/$R*$CONV, ($Vd*$Vd+$BRAT*$Vb*$Vb)/$R*$CONV, $Vo,
                   1.0/($eV*$eV+($FLOOR*$Vo)**2), sqrt($R/$CONV), $Vo*$Vo/$R*$CONV,
                   $Vd*$Vd/$R*$CONV, $Vb*$Vb/$R*$CONV];
    }
    close $fh; next unless @pts>=3;
    my $sf=$EDST{$name}/$DST{$name}; $sf=0.02 if $sf<0.02;
    my $si=$EINC{$name};             $si=0.5  if $si<0.5;
    $GAL{$name}=\@pts; $NUI{$name}=[$INC{$name},$si,$sf]; $BUL{$name}=$hasb;
    $ngal++; $npts+=@pts;
}
my @GNAMES=sort keys %GAL;
warn "galaxies used: $ngal   points: $npts\n";
# flat arrays for the converged optimizer's inner loops
my %FG;
for my $g (@GNAMES) { my %a; for my $p (@{$GAL{$g}}) { push @{$a{gg}},$p->[0]; push @{$a{gs}},$p->[1]; push @{$a{Vo}},$p->[2];
        push @{$a{w}},$p->[3]; push @{$a{sR}},$p->[4]; push @{$a{gd}},$p->[6]; push @{$a{gb}},$p->[7] } $FG{$g}=\%a }
sub two_ratio { $FREEB && $BUL{$_[0]} }       # does galaxy g carry a free bulge ratio?

# --- nu(y) evaluators ---
sub nu_simple { my $y=$_[0]; 0.5*(1.0+sqrt(1.0+4.0/$y)) }
# --- generic golden-section minimizer on [a,b] ---
my $GR=0.6180339887;
sub golden { my ($f,$a,$b,$n)=@_;
    my $c=$b-$GR*($b-$a); my $d=$a+$GR*($b-$a); my ($fc,$fd)=($f->($c),$f->($d));
    for (1..$n){ if($fc<$fd){$b=$d;$d=$c;$fd=$fc;$c=$b-$GR*($b-$a);$fc=$f->($c)}
        else{$a=$c;$c=$d;$fc=$fd;$d=$a+$GR*($b-$a);$fd=$f->($d)} }
    my $x=0.5*($a+$b); ($x,$f->($x)) }
# Brent's minimizer (Forsythe, Malcolm and Moler, fmin) on [a,b]
sub brent { my ($f,$a,$b,$tol)=@_;
    my $c=0.5*(3.0-sqrt(5.0)); my $eps=1.4901161193847656e-08;
    my $v=$a+$c*($b-$a); my $w=$v; my $x=$v; my ($d,$e)=(0,0);
    my $fx=$f->($x); my ($fv,$fw)=($fx,$fx);
    for (1..200) { my $xm=0.5*($a+$b); my $tol1=$eps*abs($x)+$tol/3.0; my $tol2=2.0*$tol1;
        last if abs($x-$xm)<=($tol2-0.5*($b-$a));
        my $gs=1;
        if (abs($e)>$tol1) { my $r=($x-$w)*($fx-$fv); my $q=($x-$v)*($fx-$fw); my $p=($x-$v)*$q-($x-$w)*$r;
            $q=2.0*($q-$r); $p=-$p if $q>0; $q=abs($q); my $et=$e; $e=$d;
            if (!(abs($p)>=abs(0.5*$q*$et) || $p<=$q*($a-$x) || $p>=$q*($b-$x))) { $d=$p/$q; my $u=$x+$d;
                $d=($xm-$x>=0?$tol1:-$tol1) if ($u-$a)<$tol2 || ($b-$u)<$tol2; $gs=0 } }
        if ($gs) { $e=($x>=$xm)?$a-$x:$b-$x; $d=$c*$e }
        my $u=(abs($d)>=$tol1)?$x+$d:$x+($d>=0?$tol1:-$tol1); my $fu=$f->($u);
        if ($fu<=$fx) { if ($u>=$x){$a=$x}else{$b=$x} ($v,$fv,$w,$fw,$x,$fx)=($w,$fw,$x,$fx,$u,$fu) }
        else { if ($u<$x){$a=$u}else{$b=$u}
            if ($fu<=$fw || $w==$x) { ($v,$fv,$w,$fw)=($w,$fw,$u,$fu) }
            elsif ($fu<=$fv || $v==$x || $v==$w) { ($v,$fv)=($u,$fu) } } }
    ($x,$fx) }
# Nelder-Mead, for the convergence rounds in two ratios
sub nm { my ($f,$x0,$st,$tol,$maxit)=@_; my $n=@$x0;
    my @s=([@$x0]); for my $j (0..$n-1){ my @x=@$x0; $x[$j]+=$st->[$j]; push @s,\@x }
    my @fv=map{$f->(@$_)}@s; my $it=0;
    while ($it++<$maxit) {
        my @o=sort{$fv[$a]<=>$fv[$b]}0..$n; @s=@s[@o]; @fv=@fv[@o];
        last if $fv[$n]-$fv[0]<$tol;
        my @c=(0)x$n; for my $k (0..$n-1){ $c[$_]+=$s[$k][$_]/$n for 0..$n-1 }
        my @xr=map{2*$c[$_]-$s[$n][$_]}0..$n-1; my $fr=$f->(@xr);
        if ($fr<$fv[0]) { my @xe=map{3*$c[$_]-2*$s[$n][$_]}0..$n-1; my $fe=$f->(@xe);
            if ($fe<$fr) { $s[$n]=\@xe; $fv[$n]=$fe } else { $s[$n]=\@xr; $fv[$n]=$fr } }
        elsif ($fr<$fv[$n-1]) { $s[$n]=\@xr; $fv[$n]=$fr }
        else { my @xc= $fr<$fv[$n] ? map{$c[$_]+0.5*($xr[$_]-$c[$_])}0..$n-1 : map{$c[$_]+0.5*($s[$n][$_]-$c[$_])}0..$n-1;
            my $fc=$f->(@xc);
            if ($fc<($fr<$fv[$n]?$fr:$fv[$n])) { $s[$n]=\@xc; $fv[$n]=$fc }
            else { for my $k (1..$n){ $s[$k]=[map{$s[0][$_]+0.5*($s[$k][$_]-$s[0][$_])}0..$n-1]; $fv[$k]=$f->(@{$s[$k]}) } } } }
    my ($ib)=sort{$fv[$a]<=>$fv[$b]}0..$n; ($fv[$ib],@{$s[$ib]}) }

# --- per-galaxy machinery ---
sub chi2_gal_q { my ($pts,$a0,$Yd,$q,$nuf)=@_; my $c2=0; my $inv_a0=1.0/$a0;
    for my $p (@$pts) { my $gbar=$p->[0]+$Yd*$p->[1]; next if $gbar<=0;
        my $gp=$nuf->($gbar*$inv_a0)*$gbar; my $d=$p->[2]-$q*sqrt($gp)*$p->[4];
        $c2+=$d*$d*$p->[3]; }
    $c2 }
# the same with the bulge free (gbar = gg + Yd gd + Yb gb)
sub chi2_gal_qb { my ($pts,$a0,$Yd,$Yb,$q,$nuf)=@_; my $c2=0; my $inv_a0=1.0/$a0;
    for my $p (@$pts) { my $gbar=$p->[0]+$Yd*$p->[6]+$Yb*$p->[7]; next if $gbar<=0;
        my $gp=$nuf->($gbar*$inv_a0)*$gbar; my $d=$p->[2]-$q*sqrt($gp)*$p->[4];
        $c2+=$d*$d*$p->[3]; }
    $c2 }
# prior cost of the best (f,i) split reproducing a given q
sub split_prior { my ($q,$ic,$si,$sf)=@_;
    my $sic=sin($ic*$D2R);
    my ($ilo,$ihi)=($ic-4.0*$si,$ic+4.0*$si);
    $ilo=10.0 if $ilo<10.0; $ihi=89.9 if $ihi>89.9;
    my $cost=sub{ my $i=$_[0]; my $f=($q*$sic/sin($i*$D2R))**2;
        (($f-1.0)/$sf)**2 + (($i-$ic)/$si)**2 };
    my (undef,$cmin)=golden($cost,$ilo,$ihi,22); $cmin }
# q bounds from the prior extremes (f in 1 +/- 4 sig_f, i in +/- 4 sig_i)
sub qbounds { my ($ic,$si,$sf)=@_; my $sic=sin($ic*$D2R);
    my $flo=1.0-4.0*$sf; $flo=0.1 if $flo<0.1; my $fhi=1.0+4.0*$sf;
    my $ilo=$ic-4.0*$si; $ilo=10.0 if $ilo<10.0;
    my $ihi=$ic+4.0*$si; $ihi=89.9 if $ihi>89.9;
    (sqrt($flo)*sin($ilo*$D2R)/$sic, sqrt($fhi)*sin($ihi*$D2R)/$sic) }

# --- the prior P(q), tabulated per galaxy from the minimum over the split ---
# The table continues split_prior's golden sequence over i to 60 steps (the same branch as its 22 steps
# when the cost is bimodal in i). P(q) is not convex everywhere: ((f-1)/sig_f)^2 with f ~ q^2 is
# concave below q = 1/sqrt3, and where the branch of i switches P has kinks. The table is cut at the kinks
# (second difference below -2000 per unit q^2) into pieces; each piece is interpolated on its own and
# searched on its own in the inner solve.
my $NQT=1000; my %PT; my $NNONCONVEX=0; my $NKINK=0; my $KCUT=-2000;
sub split_prior_exact { my ($q,$ic,$si,$sf)=@_; my $sic=sin($ic*$D2R);
    my ($ilo,$ihi)=($ic-4.0*$si,$ic+4.0*$si); $ilo=10.0 if $ilo<10.0; $ihi=89.9 if $ihi>89.9;
    my $cost=sub{ my $i=$_[0]; my $f=($q*$sic/sin($i*$D2R))**2; (($f-1.0)/$sf)**2 + (($i-$ic)/$si)**2 };
    my (undef,$c)=golden($cost,$ilo,$ihi,60); $c }
unless ($FROZEN) { for my $g (@GNAMES) { my ($ic,$si,$sf)=@{$NUI{$g}}; my ($lo,$hi)=qbounds($ic,$si,$sf);
    my $d=($hi-$lo)/$NQT; my @P=map{ split_prior_exact($lo+$_*$d,$ic,$si,$sf) } 0..$NQT;
    my @cut=(1,(0)x($NQT-1),1); my @D2=(0); for my $k (1..$NQT-1) { my $v=($P[$k-1]-2*$P[$k]+$P[$k+1])/($d*$d); push @D2,$v; $cut[$k]=1 if $v<$KCUT } push @D2,0;
    my @pc; my $k0=0; for my $k (1..$NQT) { next unless $cut[$k]; my $mn=0; for my $j ($k0+1..$k-1) { $mn=$D2[$j] if $D2[$j]<$mn } push @pc,[$k0,$k,$mn]; $k0=$k }
    my $nk=0; $nk+=$cut[$_] for 1..$NQT-1; $NKINK++ if $nk; my $cv=1; for my $pce (@pc) { $cv=0 if $pce->[2]<-1e-6 } $NNONCONVEX++ unless $cv;
    $PT{$g}={lo=>$lo,hi=>$hi,d=>$d,P=>\@P,cut=>\@cut,pc=>\@pc,convex=>$cv,nkink=>$nk} } }
sub ptab { my ($T,$q)=@_; my $x=($q-$T->{lo})/$T->{d}; my $k=int($x); $k=0 if $k<0; $k=$NQT-1 if $k>$NQT-1; my $t=$x-$k;
    my $P=$T->{P}; my $ct=$T->{cut}; my ($p0,$p1)=($P->[$k],$P->[$k+1]); my $pm=($k>0 && !$ct->[$k])?$P->[$k-1]:2*$p0-$p1; my $p2=($k+2<=$NQT && !$ct->[$k+1])?$P->[$k+2]:2*$p1-$p0;
    0.5*(2*$p0+(-$pm+$p1)*$t+(2*$pm-5*$p0+4*$p1-$p2)*$t*$t+(-$pm+3*$p0-3*$p1+$p2)*$t*$t*$t) }

# --- the converged per-galaxy optimizer ---
my ($ULO,$UHI)=(log10(0.03),log10(10.0)); my $NGU=50; my $NG2=20; our $QTOL=1e-10;
my %STAT=(evals=>0,maxround=>0,capped=>0,bound=>0,fits=>0); my %BOUNDG;
# sufficient statistics of chi2(q) = Soo - 2 q Sop + q^2 Spp at fixed ratios
sub suff { my ($g,$ia,$Yd,$Yb,$nuf)=@_; my $G=$FG{$g}; my ($gg,$Vo,$w,$sR)=@{$G}{qw(gg Vo w sR)};
    my ($oo,$op,$pp,$n)=(0,0,0,0);
    if (defined $Yb) { my ($gd,$gb)=@{$G}{qw(gd gb)};
        for my $k (0..$#$gg) { my $gbar=$gg->[$k]+$Yd*$gd->[$k]+$Yb*$gb->[$k]; next if $gbar<=0;
            my $vp=sqrt($nuf->($gbar*$ia)*$gbar)*$sR->[$k]; my $wv=$w->[$k]*$vp; my $vo=$Vo->[$k];
            $oo+=$w->[$k]*$vo*$vo; $op+=$wv*$vo; $pp+=$wv*$vp; $n++ } }
    else { my $gs=$G->{gs};
        for my $k (0..$#$gg) { my $gbar=$gg->[$k]+$Yd*$gs->[$k]; next if $gbar<=0;
            my $vp=sqrt($nuf->($gbar*$ia)*$gbar)*$sR->[$k]; my $wv=$w->[$k]*$vp; my $vo=$Vo->[$k];
            $oo+=$w->[$k]*$vo*$vo; $op+=$wv*$vo; $pp+=$wv*$vp; $n++ } }
    ($oo,$op,$pp,$n) }
# objective at log10 ratios (u, v) with q profiled exactly; returns (F, q)
sub fobj { my ($g,$ia,$nuf,$u,$v)=@_; $STAT{evals}++;
    my $two=two_ratio($g); my $Yd=10**$u;
    my ($oo,$op,$pp)=suff($g,$ia,$Yd,($two?10**$v:undef),$nuf);
    my $f=(($u-$LGY0)/$PRIOR)**2; $f+=(($v-$LGYB0)/$PRIORB)**2 if $two;
    return ($f+$oo-2*$op+$pp,1.0) if $FROZEN;
    my $T=$PT{$g}; my $h=sub{ my $q=$_[0]; $pp*$q*$q-2*$op*$q+ptab($T,$q) };
    my ($q,$hq)=(undef,1e300);
    for my $pce (@{$T->{pc}}) { my ($qa,$qb)=($T->{lo}+$pce->[0]*$T->{d},$T->{lo}+$pce->[1]*$T->{d}); my ($x,$hx);
        if (2*$pp+$pce->[2]>0) { ($x,$hx)=brent($h,$qa,$qb,$QTOL) }          # h convex on this piece
        else { my ($bk,$bv)=(0,1e300); for my $k (0..20) { my $qq=$qa+($qb-$qa)*$k/20; my $hv=$h->($qq); ($bk,$bv)=($k,$hv) if $hv<$bv }
            my $st=($qb-$qa)/20; ($x,$hx)=brent($h,$qa+($bk>0?$bk-1:0)*$st,$qa+($bk<20?$bk+1:20)*$st,$QTOL) }
        ($q,$hq)=($x,$hx) if $hx<$hq;
        for my $qe ($qa,$qb) { my $he=$h->($qe); ($q,$hq)=($qe,$he) if $he<$hq } }
    ($f+$oo+$hq,$q) }
# Newton with finite differences and a backtracking line search, inside the box
sub newton2 { my ($F,$x,$y)=@_; my $h=1e-4; my $f0=$F->($x,$y);
    for my $it (1..40) {
        my ($fp0,$fm0,$f0p,$f0m)=($F->($x+$h,$y),$F->($x-$h,$y),$F->($x,$y+$h),$F->($x,$y-$h));
        my ($fpp,$fpm,$fmp,$fmm)=($F->($x+$h,$y+$h),$F->($x+$h,$y-$h),$F->($x-$h,$y+$h),$F->($x-$h,$y-$h));
        my $gx=($fp0-$fm0)/(2*$h); my $gy=($f0p-$f0m)/(2*$h);
        my $hxx=($fp0-2*$f0+$fm0)/($h*$h); my $hyy=($f0p-2*$f0+$f0m)/($h*$h); my $hxy=($fpp-$fpm-$fmp+$fmm)/(4*$h*$h);
        my $det=$hxx*$hyy-$hxy*$hxy; my ($dx,$dy);
        if ($hxx>0 && $hyy>0 && $det>0) { $dx=-($hyy*$gx-$hxy*$gy)/$det; $dy=-($hxx*$gy-$hxy*$gx)/$det }
        else { my $gn=sqrt($gx*$gx+$gy*$gy)||1; ($dx,$dy)=(-0.02*$gx/$gn,-0.02*$gy/$gn) }
        my $sl=sqrt($dx*$dx+$dy*$dy); if ($sl>0.3) { $dx*=0.3/$sl; $dy*=0.3/$sl }
        my ($t,$fn)=(1,undef);
        while ($t>1e-8) { $fn=$F->($x+$t*$dx,$y+$t*$dy); last if $fn<$f0; $t*=0.5 }
        last unless defined $fn && $fn<$f0;
        my $gain=$f0-$fn; ($x,$y,$f0)=($x+$t*$dx,$y+$t*$dy,$fn);
        last if $gain<1e-12 }
    ($f0,$x,$y) }
# the objective at given nuisances, with split_prior (the value carried forward)
sub rec_obj { my ($g,$a0,$nuf,$Yd,$Yb,$q)=@_; my ($ic,$si,$sf)=@{$NUI{$g}};
    my $f = two_ratio($g) ? chi2_gal_qb($GAL{$g},$a0,$Yd,$Yb,$q,$nuf)+((log10($Yb)-$LGYB0)/$PRIORB)**2
                          : chi2_gal_q($GAL{$g},$a0,$Yd,$q,$nuf);
    $f += ((log10($Yd)-$LGY0)/$PRIOR)**2;
    $f += split_prior($q,$ic,$si,$sf) unless $FROZEN;
    $f }
# profile (Yd[,Yb],q) for one galaxy at fixed a0/shape; returns (chi2+priors, Yd, q, Yb, rounds, npts used)
# $dense > 1 multiplies the coarse grid density (section L's re-check)
sub fit_gal { my ($g,$a0,$nuf,$dense)=@_; $dense||=1; my $ia=1.0/$a0; my ($fb,$bu,$bv,$rounds);
    if (!two_ratio($g)) {
        my $F=sub{ (fobj($g,$ia,$nuf,$_[0]))[0] }; my $ng=$NGU*$dense;
        my @u=map{$ULO+($UHI-$ULO)*$_/$ng}0..$ng; my @f; { local $QTOL=1e-6; @f=map{$F->($_)}@u }
        my @lm=grep{ ($_==0||$f[$_]<=$f[$_-1]) && ($_==$ng||$f[$_]<=$f[$_+1]) } 0..$ng;
        $fb=1e300;
        for my $j (@lm) { my ($x,$fx)=brent($F,$u[$j>0?$j-1:0],$u[$j<$ng?$j+1:$ng],1e-10);
            ($x,$fx)=($u[$j],$f[$j]) if $f[$j]<$fx; ($fb,$bu)=($fx,$x) if $fx<$fb }
        $rounds=1; my $hw=($UHI-$ULO)/$ng/4;
        while ($rounds<$CAP) { $rounds++; my $a=$bu-$hw; $a=$ULO if $a<$ULO; my $b=$bu+$hw; $b=$UHI if $b>$UHI;
            my ($x,$fx)=brent($F,$a,$b,1e-12); my $gain=$fb-$fx; ($fb,$bu)=($fx,$x) if $fx<$fb; $hw*=0.5; last if $gain<$TOL }
    } else {
        my $F=sub{ my ($u,$v)=@_; return 1e30 if $u<$ULO||$u>$UHI||$v<$ULO||$v>$UHI; (fobj($g,$ia,$nuf,$u,$v))[0] };
        my $ng=$NG2*$dense; my @u=map{$ULO+($UHI-$ULO)*$_/$ng}0..$ng; my @f;
        { local $QTOL=1e-6; for my $i (0..$ng) { for my $j (0..$ng) { $f[$i][$j]=$F->($u[$i],$u[$j]) } } }
        my @lm; for my $i (0..$ng) { for my $j (0..$ng) { my $ok=1;
            for my $di (-1,0,1) { for my $dj (-1,0,1) { next if !$di && !$dj; my ($a,$b)=($i+$di,$j+$dj);
                next if $a<0||$b<0||$a>$ng||$b>$ng; $ok=0 if $f[$a][$b]<$f[$i][$j] } } push @lm,[$i,$j] if $ok } }
        @lm=(sort{$f[$a->[0]][$a->[1]]<=>$f[$b->[0]][$b->[1]]}@lm)[0..($#lm<2?$#lm:2)];
        $fb=1e300;
        for my $m (@lm) { my ($x,$y)=($u[$m->[0]],$u[$m->[1]]); for ($x,$y) { $_=$ULO+2e-4 if $_<$ULO+2e-4; $_=$UHI-2e-4 if $_>$UHI-2e-4 }
            my ($fx,$xx,$yy)=newton2($F,$x,$y); ($fb,$bu,$bv)=($fx,$xx,$yy) if $fx<$fb }
        $rounds=1;
        while ($rounds<$CAP) { $rounds++; my ($fn,$x,$y)=nm($F,[$bu,$bv],[0.01,0.01],1e-10,400);
            ($fn,$x,$y)=newton2($F,$x,$y) if $fn<$fb;
            my $gain=$fb-$fn; ($fb,$bu,$bv)=($fn,$x,$y) if $fn<$fb; last if $gain<$TOL }
    }
    $STAT{fits}++; $STAT{maxround}=$rounds if $rounds>$STAT{maxround}; $STAT{capped}++ if $rounds>=$CAP;
    my ($fq,$q)=fobj($g,$ia,$nuf,$bu,$bv);
    my $Yd=10**$bu; my $Yb=two_ratio($g)?10**$bv:$BRAT*$Yd;
    unless ($FROZEN) { # refine q with split_prior itself (the table only guides the search)
        my ($ic,$si,$sf)=@{$NUI{$g}}; my ($oo,$op,$pp)=suff($g,$ia,$Yd,(two_ratio($g)?$Yb:undef),$nuf); my $T=$PT{$g};
        my $hr=sub{ my $x=$_[0]; $pp*$x*$x-2*$op*$x+split_prior($x,$ic,$si,$sf) };
        my $a=$q-2*$T->{d}; $a=$T->{lo} if $a<$T->{lo}; my $b=$q+2*$T->{d}; $b=$T->{hi} if $b>$T->{hi};
        my $h0=$hr->($q); my ($x,$hx)=brent($hr,$a,$b,1e-12); $q=$x if $hx<$h0;
        for my $qe ($a,$b) { my $he=$hr->($qe); if ($he<($hx<$h0?$hx:$h0)) { $q=$qe; $hx=$he } } }
    my $bnd = (abs($bu-$ULO)<1e-6||abs($bu-$UHI)<1e-6) ? 1 : 0;
    $bnd=1 if two_ratio($g) && (abs($bv-$ULO)<1e-6||abs($bv-$UHI)<1e-6);
    $bnd=1 if !$FROZEN && (abs($q-$PT{$g}{lo})<1e-9||abs($q-$PT{$g}{hi})<1e-9);
    $STAT{bound}+=$bnd; if ($bnd) { my $w=''; $w.='Yd ' if abs($bu-$ULO)<1e-6||abs($bu-$UHI)<1e-6; $w.='Yb ' if two_ratio($g) && (abs($bv-$ULO)<1e-6||abs($bv-$UHI)<1e-6);
        $w.=(abs($q-$PT{$g}{lo})<1e-9?'q_lo':'q_hi') if !$FROZEN && (abs($q-$PT{$g}{lo})<1e-9||abs($q-$PT{$g}{hi})<1e-9); $w=~s/\s+$//; $BOUNDG{$g}=$w }
    my $n=0; for my $p (@{$GAL{$g}}) { my $gb = two_ratio($g) ? $p->[0]+$Yd*$p->[6]+$Yb*$p->[7] : $p->[0]+$Yd*$p->[1]; $n++ if $gb>0 }
    my $best=rec_obj($g,$a0,$nuf,$Yd,$Yb,$q);
    ($best,$Yd,$q,$Yb,$rounds,$n,$bnd) }
# gbar of a point at a galaxy's fitted ratios
sub gbar_pt { my ($g,$p,$Yd,$Yb)=@_; two_ratio($g) ? $p->[0]+$Yd*$p->[6]+$Yb*$p->[7] : $p->[0]+$Yd*$p->[1] }
# ---- end of the functions copied from two_constant_fit.pl ----

# --- per-galaxy stats, aggregated by Q class ---
my %CL=(1=>{},2=>{});
my $inv_a0=1.0/$A0;
for my $g (@GNAMES) {
    my (undef,$Yd,$q,$Yb)=fit_gal($g,$A0,\&nu_simple); my $q2=$q*$q;
    my ($c2,$n)=(0,0); my (@rall,@rdeep,@rhigh);
    for my $p (@{$GAL{$g}}) { my $gbar=gbar_pt($g,$p,$Yd,$Yb); next if $gbar<=0;
        my $gp=nu_simple($gbar*$inv_a0)*$gbar;
        my $d=$p->[2]-$q*sqrt($gp)*$p->[4]; $c2+=$d*$d*$p->[3]; $n++;
        my $r=log10($p->[5]/$q2)-log10($gp);
        push @rall,$r; if ($gbar*$inv_a0<1.0){push @rdeep,$r}else{push @rhigh,$r} }
    next unless $n>=3;
    my $rms=sub{ my @x=@_; return undef unless @x>=3;
        my $m=0; $m+=$_ for @x; $m/=@x; my $s=0; $s+=($_-$m)**2 for @x; sqrt($s/@x) };
    my $cl=$CL{$Q{$g}};
    push @{$cl->{chin}},$c2/$n;
    push @{$cl->{rms}},$rms->(@rall);
    my ($rd,$rh)=($rms->(@rdeep),$rms->(@rhigh));
    push @{$cl->{deep}},@rdeep; push @{$cl->{high}},@rhigh;
    push @{$cl->{ratio}},$rd/$rh if defined $rd && defined $rh && $rh>0;
    push @{$cl->{gal}},[ (defined $rd && defined $rh && $rh>0) ? $rd/$rh : undef, \@rdeep, \@rhigh ];
}
sub med { my @x=sort{$a<=>$b}@_; @x ? $x[int(@x/2)] : undef }
sub prms { my @x=@_; my $m=0; $m+=$_ for @x; $m/=@x;
    my $s=0; $s+=($_-$m)**2 for @x; (sqrt($s/@x),scalar @x) }
printf "machinery: the per-galaxy fit of two_constant_fit.pl (converged; bulge %s); MOND's simple function at a0 = %.4e\n\n",$FREEB?'free per galaxy':'fixed at 1.4 x disk',$A0;
printf "%-28s %10s %10s\n",'','Q=1','Q=2';
my %out;
for my $q (1,2) { my $c=$CL{$q};
    $out{$q}{ngal}=scalar @{$c->{chin}};
    $out{$q}{chin}=med(@{$c->{chin}});
    $out{$q}{rms}=med(@{$c->{rms}});
    ($out{$q}{pdeep},$out{$q}{ndeep})=prms(@{$c->{deep}});
    ($out{$q}{phigh},$out{$q}{nhigh})=prms(@{$c->{high}});
    $out{$q}{ratio}=med(@{$c->{ratio}});
}
printf "%-28s %10d %10d\n",'galaxies',map{$out{$_}{ngal}}1,2;
printf "%-28s %10.2f %10.2f\n",'median chi2/N',map{$out{$_}{chin}}1,2;
printf "%-28s %10.4f %10.4f\n",'median per-gal RAR rms (dex)',map{$out{$_}{rms}}1,2;
printf "%-28s %10.4f %10.4f\n",'pooled rms, deep (y<1)',map{$out{$_}{pdeep}}1,2;
printf "%-28s %10d %10d\n",'  deep points',map{$out{$_}{ndeep}}1,2;
printf "%-28s %10.4f %10.4f\n",'pooled rms, high (y>=1)',map{$out{$_}{phigh}}1,2;
printf "%-28s %10d %10d\n",'  high points',map{$out{$_}{nhigh}}1,2;
printf "%-28s %10.3f %10.3f\n",'median per-gal deep/high',map{$out{$_}{ratio}}1,2;
print "\ndiscriminant: slow re-equilibration would give Q=2 a larger deep/high than Q=1;\n";
print "noise from poorer data inflates Q=2 at every y alike.\n";
# galaxy-bootstrap errors of the discriminant, resampling the galaxies within each class
{ my $seed=20261010; my $rng=sub{ $seed=($seed*1103515245+12345)%2147483648; $seed/2147483648 };
  my $NB=4000; my (@rr,@rp);
  my %G=map{ $_=>$CL{$_}{gal} } (1,2);
  for (1..$NB) { my %m; my %pd; my %ph;
      for my $q (1,2) { my $L=$G{$q}; my @r; my (@d,@h);
          for (1..@$L) { my $e=$L->[int($rng->()*@$L)]; push @r,$e->[0] if defined $e->[0]; push @d,@{$e->[1]}; push @h,@{$e->[2]} }
          $m{$q}=med(@r); ($pd{$q})=prms(@d); ($ph{$q})=prms(@h) }
      push @rr,$m{2}/$m{1}; push @rp,($pd{2}/$ph{2})/($pd{1}/$ph{1}) }
  my $sd=sub{ my @x=@_; my $m=0; $m+=$_ for @x; $m/=@x; my $v=0; $v+=($_-$m)**2 for @x; sqrt($v/(@x-1)) };
  my $pc=sub{ my @x=sort{$a<=>$b}@_; ($x[int(0.16*@x)],$x[int(0.84*@x)],$x[int(0.025*@x)],$x[int(0.975*@x)]) };
  my $r0=$out{2}{ratio}/$out{1}{ratio}; my $p0=($out{2}{pdeep}/$out{2}{phigh})/($out{1}{pdeep}/$out{1}{phigh});
  my @a=$pc->(@rr); my @b=$pc->(@rp);
  printf "\ngalaxy bootstrap (%d draws within each class):\n",$NB;
  printf "  ratio of the median per-galaxy deep/high, Q=2 over Q=1: %.3f, SD %.3f, 16-84 %.3f to %.3f, 2.5-97.5 %.3f to %.3f; fraction of draws <= 1: %.4f\n",
      $r0,$sd->(@rr),@a,(grep{$_<=1}@rr)/$NB;
  printf "  ratio of the pooled deep/high rms, Q=2 over Q=1: %.3f, SD %.3f, 16-84 %.3f to %.3f, 2.5-97.5 %.3f to %.3f; fraction of draws <= 1: %.4f\n",
      $p0,$sd->(@rp),@b,(grep{$_<=1}@rp)/$NB; }
print "done.\n";
