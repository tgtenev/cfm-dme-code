#!/usr/bin/perl
# two_constant_fit.pl: fits of the 153 SPARC rotation curves with each galaxy's distance and inclination profiled
# under their catalog priors, for MOND's simple and standard functions, the empirical function and the square-root
# form, and the two-constant fit of the boost law, in which the free volume and the scale are both free.
#
# Papers I and II use these numbers:
#   Paper I §4.1: the median fitted shifts of the interpolation functions, f = 0.985 to 1.002 and
#     Delta i = -0.15 to -0.10 degrees (section A, two_constant_fit-results.txt lines 5 to 9).
#   Paper I §4.2 and Table 2, Paper II §2 and §7.1: the two-constant minimum, delta_fv = 0.295,
#     a0 = 0.92e-10 m/s^2 and chi^2 = 4319.2 (section B, lines 38 and 60); the product D ln(1 + delta_fv) there,
#     1.52 +/- 0.60 at H0 = 67.4 km/s/Mpc and 1.64 +/- 0.65 at 73 (sections H and M, see the conversion below);
#     the fitted onset of the square-root form, X_sat = 3.30 (section E, line 115). The clustered error of X_sat
#     (0.9) and the comparisons with the constrained fit (3.9 +/- 9.7, 5.1 +/- 10.7, 9.1 +/- 8.8) come from
#     constrained_fit.py.
#   Paper I Figures 5 and 6: Figures 5(a) and 6 draw the two-constant fit from section B of
#     two_constant_fit-results.txt, the node table (delta_fv, a0 with its stat error, chi^2 at the fitted a0;
#     lines 42 to 57) and the minimum (lines 38 and 60); their constrained curves come from
#     constrained_fit-valley.csv. Figure 5(b) uses X_sat = 3.30 from section E. The figures and their drawing
#     script are not in the deposit.
# The other sections are diagnostics and checks. refit_core.py ports this machinery to numpy for constrained_fit.py
# and the AQUAL refits; aqual-refit-fit.py --part machinery checks the port against section B of
# two_constant_fit-results.txt and against two_constant_fit-frozen-results.txt. The sixteen treatments of
# Paper I §4.1 are computed by constrained_fit.py.
#
# Laws, by the codes used in the code and in the printed labels (y = g_bar/a0, X = g/a0):
#   simple      MOND's simple function, nu(y) = 1/2 + sqrt(1/4 + 1/y).
#   standard    MOND's standard function, mu(X) = X/sqrt(1 + X^2).
#   bcf         a sharp-knee function, mu(X) = X tanh(1/X).
#   T11         the empirical function of McGaugh, Lelli and Schombert, nu(y) = 1/(1 - exp(-sqrt(y))).
#   T3          the boost law of Paper I, Eq. (8), mu(X) = (e^(lambda X) - 1)/(e^(lambda X) - 1 + delta_fv) with
#               lambda = ln(1 + delta_fv). Scanned over delta_fv with a0 refitted at each node, it is the
#               two-constant fit of Paper I §4.2.
#   sqrt-Xsat5  the square-root form of Paper I §4.2, mu = b/(1 + b) with b = X sqrt(1 + (X/X_sat)^2), at X_sat = 5.
#   sqrt-Xsat   the same form with X_sat fitted (section E).
# Laws given by mu are tabulated as nu(y) on a grid in log10 y and interpolated.
#
# The divisor and the product. D in the outputs is c H0/a0 at the H0 printed beside it (67.4, 70 or 73 km/s/Mpc),
# and the product is (c H0/a0) ln(1 + delta_fv). Paper I's divisor, Eq. (10), is c H_dS/a0 with
# H_dS = sqrt(Omega_Lambda) H0 = 0.8275 H0 (Omega_Lambda = 0.6847). Paper I's divisor and product are therefore
# 0.8275 times the printed ones, and so are every mean, SD and percentile of them, since the factor carries no error.
# At the two-constant minimum the printed products 1.832 (67.4) and 1.985 (73) give Paper I's 1.52 and 1.64, and
# their clustered SDs 0.7217 and 0.7817 over the bootstrap draws (section M, lines 253 and 254) give its 0.60 and 0.65.
# delta_fv = 0.1725, where 1/ln(1 + delta_fv) = 2 pi, is printed as a reference point of the scan.
#
# The fit (Paper I §4.1). Each point weighs 1/(eV^2 + (0.05 V_obs)^2). A distance factor f = D/D_cat scales
# luminosity as D^2 and radius as D, so g_bar = V_bar^2/R is unchanged point by point and V_pred scales as sqrt(f).
# An inclination i scales V_obs, eV and the floor by sin(i_cat)/sin(i). Dividing each residual by that factor, both
# nuisances enter chi^2 through one factor q (Paper I Eqs. (14) and (15)):
#   chi2_pt = (V_obs - q V_pred)^2/sigma_pt^2,   q = sqrt(f) sin(i)/sin(i_cat),
#   P(q) = min over i of [((f - 1)/sig_f)^2 + ((i - i_cat)/sig_i)^2],   f = (q sin(i_cat)/sin(i))^2,
# with sig_f = e_D/D (at least 0.02) and sig_i = e_Inc (at least 0.5 degrees), searched within 4 catalog sigma.
# The disk ratio has a lognormal prior of 0.10 dex about 0.5, and each bulge ratio, free per galaxy, one of 0.10 dex
# about 0.7 (Li et al. 2018). The sample: SPARC galaxies with quality flag not 3, inclination at least 30 degrees
# and at least 3 points, 153 galaxies with 3,168 points, of which 31 have a bulge.
#
# The optimizer. chi^2 is quadratic in q at fixed mass-to-light ratios, so q is profiled exactly at every trial:
# q* minimizes Spp q^2 - 2 Sop q + P(q) on [q_lo, q_hi] (Brent, and both bounds), with P(q) tabulated per galaxy on
# 1000 intervals (cubic interpolation) from a 60-step golden search over i. The ratios are searched on a coarse grid
# over the box Y in [0.03, 10] (51 nodes for one ratio, 21 x 21 for two), every local minimum of the grid is refined
# (Brent in one ratio, Newton with a line search in two), and rounds restart the local search from the optimum
# (Brent on a narrowed bracket, Nelder-Mead in two ratios) until a round lowers the galaxy's objective by less than
# --tol; the cap --cap is never reached (section L prints the largest count). The value carried forward is the
# objective chi^2 + priors + P(q), with P(q) from a 22-step golden search (split_prior), at the optimum.
# The global a0 of a law, and of each node of a scanned law, is fitted by a parabola in log10 a0 through a ladder of
# converged totals (+/-0.02 dex, vertex steps, then vertex +/-0.01 dex), which also gives its Delta chi^2 = 1 error
# with the nuisances refitted. A scanned law's minimum is the parabola through its three lowest nodes.
# Errors clustered by galaxy (sections F and M) come from --nboot galaxy-bootstrap draws, each refitting every law
# along its stored per-galaxy profile: a parabola in log10 a0 at each node and the three-lowest-node parabola across
# the nodes. The delta-method error at the best fits is printed beside them. Sums over galaxies run in sorted name order.
#
# Output sections, in the order printed (each headed by its letter):
#   R  reference fits with distances and inclinations at catalog values (simple, sqrt-Xsat5, T3), marginalized run only.
#   A  the fixed laws: a0, chi^2, scatter, median Y_d, median fitted f and i - i_cat, D; then a0 with its stat error.
#   B  the two-constant fit: T3 over a grid of delta_fv with a0 refitted at each node, the minimum and the node table.
#   D  binned mean residuals of the simple function in log10 y; D2, of T3 at the minimum and of the standard function
#      in log10 g_bar.
#   E  the square-root form: chi^2 over X_sat with a0 refitted at each node, and the minimum.
#   F  Delta chi^2 between the laws with galaxy-clustered errors, and the clustered errors of a0, delta_fv and X_sat.
#   G  the fitted scales and D.
#   H  the product at the minimum and the quantities derived from delta_fv, among them the onset X_onset = xi*/lambda
#      of Paper I §3.5 and the amplification A = 2/(Z1 delta_fv^2) of Paper II Eq. (4) at Z1 = 1.
#   I  the bands 4 <= y <= 8 and y > 8 (diagnostics, not in chi^2).
#   J  scatter, mass-to-light spreads, points used and chi^2 per degree of freedom at the best fits.
#   K  the T3 scan at the wide mass-to-light prior (--wide), marginalized run only.
#   N  the knee fit: the simple function on the points above y = 2 alone, catalog-distance run only.
#   L  convergence of the optimizer, a re-check on grids twice as dense, the brute-force grid check (--verify).
#   M  the product over the bootstrap draws of section F.
#
# Options (--key=value, anywhere on the command line):
#   --bulge=free|fixed   a free bulge ratio per galaxy (default), or Y_b = 1.4 Y_d
#   --frozen=1           distances and inclinations at catalog values (q = 1)
#   prior_dex            the third positional argument: the width of both mass-to-light priors (default 0.10 dex);
#                        --priorb=dex sets the bulge prior's width alone
#   --wide=dex           the prior of section K (default 0.25; 0 skips the section)
#   --extra-grid=list    delta_fv nodes added to the grid 0.05 to 0.30 (default 0.35,0.40,0.45)
#   --onset-grid=list    the X_sat nodes of section E
#   --verify=G1,G2,...   the galaxies of the brute-force grid check of section L
#   --nboot, --seed      the bootstrap draws (2000) and their seed (20261010)
#   --tol, --cap         the optimizer's tolerance (1e-7) and round cap (50)
#   --recheck=0          skips the re-check of section L on grids twice as dense
#   --sections=letters   runs the given sections only (default RABDEFGHIJKLMN)
#   --test-ngal=N, --test-gals=G1,G2   a quick check on the first N galaxies or on the named ones, never for results
#
# Inputs: data/sparc/SPARC_Lelli2016c.mrt (SPARC Table 1: distances, inclinations, their errors, quality flags) and
# the mass models in data/sparc/rotmod_ltg/. Output: the results on standard output, progress on standard error.
# Run from the repository root:
#   perl rotation-curves/two_constant_fit.pl data/sparc/SPARC_Lelli2016c.mrt data/sparc/rotmod_ltg \
#        --extra-grid=0.35,0.40,0.45,0.50,0.55,0.60 --verify=UGC06787,NGC5371,UGC02953,IC4202,UGC09133,NGC7331 \
#        > rotation-curves/two_constant_fit-results.txt
#   perl rotation-curves/two_constant_fit.pl data/sparc/SPARC_Lelli2016c.mrt data/sparc/rotmod_ltg \
#        --frozen=1 --verify=UGC06787,NGC5371,UGC02953 > rotation-curves/two_constant_fit-frozen-results.txt
# Run times: about 87 minutes for the first (5195 s) and 22 minutes for the second (1321 s), printed in section L.
use strict; use warnings; use POSIX qw(tanh);
$| = 1;

my $CONV=3.24078e-14; my $FLOOR=0.05; my $Y0=0.5; my $BRAT=1.4; our $PRIOR=0.10;
my $C=2.99792458e8; my $MPC=3.0857e22; my $PI=4*atan2(1,1); my $D2R=$PI/180.0;
my $INV_LN10=1.0/log(10.0); sub log10 { log($_[0])*$INV_LN10 } my $LGY0=log10($Y0);
my $YB0=0.7; my $LGYB0=log10($YB0);          # Li et al. 2018 bulge prior centre

# --- options: --key=value anywhere on the command line ---
my %OPT=(bulge=>'free',frozen=>0,priorb=>'',tol=>1e-7,cap=>50,nboot=>2000,seed=>20261010,
         sections=>'',verify=>'',wide=>0.25,onset_grid=>'2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0',
         extra_grid=>'0.35,0.40,0.45',test_ngal=>0,test_gals=>'',recheck=>1);
my @POS;
for my $a (@ARGV) { if ($a=~/^--([\w-]+)=(.*)$/) { (my $k=$1)=~tr/-/_/; die "unknown option --$1\n" unless exists $OPT{$k}; $OPT{$k}=$2 }
                    else { push @POS,$a } }
my ($t1path,$rotdir,$priorarg,@EXTRA)=@POS;
die "usage: perl rotation-curves/two_constant_fit.pl SPARC_Lelli2016c.mrt rotmod_ltg [prior_dex] [--key=value ...]\n"
    if !defined $rotdir || @EXTRA;
$PRIOR=$priorarg if $priorarg;
our $PRIORB = $OPT{priorb} ne '' ? $OPT{priorb} : $PRIOR;
our $FROZEN = $OPT{frozen} ? 1 : 0;
my $FREEB = $OPT{bulge} eq 'free' ? 1 : 0;
die "--bulge must be free or fixed\n" unless $OPT{bulge}=~/^(free|fixed)$/;
my $TOL=$OPT{tol}; my $CAP=$OPT{cap};
my @GRID=(0.05,0.08,0.10,0.12,0.14,0.16,0.1725,0.20,0.24,0.30);   # the delta_fv grid of section B, before --extra-grid
my %SEC=map{$_=>1} split //,($OPT{sections} || 'RABDEFGHIJKLMN');
warn "prior=$PRIOR\n";
warn "bulge=$OPT{bulge} priorb=$PRIORB frozen=$FROZEN tol=$TOL cap=$CAP sections=".join('',sort keys %SEC)."\n";

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
if ($OPT{test_ngal} || $OPT{test_gals}) { @GNAMES= $OPT{test_gals} ? (grep{ $GAL{$_} } split /,/,$OPT{test_gals}) : @GNAMES[0..$OPT{test_ngal}-1]; my %k=map{$_=>1}@GNAMES;
    for (keys %GAL){ delete $GAL{$_} unless $k{$_} } $ngal=@GNAMES; $npts=0; $npts+=@{$GAL{$_}} for @GNAMES;
    warn "TEST MODE: $ngal galaxies (never used for results)\n" }
my $NBUL=scalar(grep{$BUL{$_}}@GNAMES);
warn "galaxies used: $ngal   points: $npts\n";
# flat arrays for the converged optimizer's inner loops
my %FG;
for my $g (@GNAMES) { my %a; for my $p (@{$GAL{$g}}) { push @{$a{gg}},$p->[0]; push @{$a{gs}},$p->[1]; push @{$a{Vo}},$p->[2];
        push @{$a{w}},$p->[3]; push @{$a{sR}},$p->[4]; push @{$a{gd}},$p->[6]; push @{$a{gb}},$p->[7] } $FG{$g}=\%a }
sub two_ratio { $FREEB && $BUL{$_[0]} }       # does galaxy g carry a free bulge ratio?

# --- nu(y) evaluators ---
sub nu_simple { my $y=$_[0]; 0.5*(1.0+sqrt(1.0+4.0/$y)) }
sub solve_X { my ($mu,$y)=@_; my ($lo,$hi)=(1e-10,1e10);
    for (1..70){my $m=sqrt($lo*$hi); if($m*$mu->($m)<$y){$lo=$m}else{$hi=$m}} sqrt($lo*$hi) }
my $LY0=-9.0; my $STEP=0.02; my $INVSTEP=1.0/$STEP; my $NT=int((9.0-$LY0)/$STEP);
# a law given by mu(X): log10 nu(y) tabulated in log10 y from -9 to 9 in steps of 0.02, X solving X mu(X) = y
sub build_table { my $mu=$_[0]; my @lnu;
    for (my $i=0;$i<=$NT;$i++){my $y=10**($LY0+$i*$STEP); push @lnu,log10(solve_X($mu,$y)/$y)}
    \@lnu }
# the boost law (T3), Paper I Eq. (8)
sub mu_boost { my ($X,$del,$lam)=@_; my $u=$lam*$X;
    return 1.0 if $u>40.0; my $E=exp($u)-1.0; $E/($E+$del) }
# nu(y) from a table, linear in log10 nu between the nodes
sub tabf { my $t=$_[0]; sub { my $x=(log($_[0])*$INV_LN10-$LY0)*$INVSTEP; my $i=int($x);
    $i=0 if $i<0; $i=$NT-1 if $i>=$NT; my $f=$x-$i; 10**($t->[$i]*(1-$f)+$t->[$i+1]*$f) } }

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

# --- totals, the a0 fit, storage for the clustered errors ---
my %EVS;   # $EVS{law}{node} = { x=>shape value, la=>{ sprintf('%.6f',la) => [per-galaxy F, sorted names] } }
my %BESTFIT; # $BESTFIT{law}{node} = { a0, la, chi2, per=>{g=>[F,Yd,q,Yb,n]} , sig_la, ymin }
sub total_chi2 { my ($a0,$nuf,$dense)=@_; my $tot=0; my (%bY,%bq,%bB,%bc,%bn);
    for my $g (@GNAMES){ my ($c2,$Yd,$q,$Yb,undef,$n)=fit_gal($g,$a0,$nuf,$dense);
        $tot+=$c2; $bY{$g}=$Yd; $bq{$g}=$q; $bB{$g}=$Yb; $bc{$g}=$c2; $bn{$g}=$n }
    ($tot,\%bY,\%bq,\%bB,\%bc,\%bn) }
sub lsq { my ($X,$Y)=@_; my $k=@{$X->[0]}; my @M; my @v;
    for my $i (0..$k-1){ for my $j (0..$k-1){ my $s=0; $s+=$_->[$i]*$_->[$j] for @$X; $M[$i][$j]=$s }
        my $s=0; for my $r (0..$#$X){ $s+=$X->[$r][$i]*$Y->[$r] } $v[$i]=$s }
    for my $i (0..$k-1){ my $p=$i; for my $r ($i+1..$k-1){ $p=$r if abs($M[$r][$i])>abs($M[$p][$i]) }
        @M[$i,$p]=@M[$p,$i]; @v[$i,$p]=@v[$p,$i]; my $d=$M[$i][$i]; return undef if abs($d)<1e-300;
        for my $j (0..$k-1){ $M[$i][$j]/=$d } $v[$i]/=$d;
        for my $r (0..$k-1){ next if $r==$i; my $f=$M[$r][$i]; next unless $f;
            for my $j (0..$k-1){ $M[$r][$j]-=$f*$M[$i][$j] } $v[$r]-=$f*$v[$i] } }
    \@v }
# least-squares parabola over (x,y) points within $win of the lowest; returns (xmin, ymin, sigma_x, rms) or undef
# (solved in the centred abscissa t = x - x0, x0 the abscissa of the lowest point: on the raw abscissa, log10 a0
#  near -10, the normal equations lose their digits when two ladder keys lie 1e-4 dex apart)
sub parab_ls { my ($xs,$ys,$win)=@_; my ($lo)=sort{$a<=>$b}@$ys; my @u=grep{$ys->[$_]<=$lo+$win}0..$#$xs;
    @u=(sort{$ys->[$a]<=>$ys->[$b]}0..$#$xs)[0..2] if @u<3; return undef if @u<3;
    my ($i0)=sort{$ys->[$a]<=>$ys->[$b]}@u; my $x0=$xs->[$i0]; my @t=map{$xs->[$_]-$x0}@u;
    my $c=lsq([map{[1,$_,$_*$_]}@t],[@{$ys}[@u]]); return undef unless $c && $c->[2]>0;
    my $tm=-$c->[1]/(2*$c->[2]); my $rr=0; $rr+=($ys->[$u[$_]]-($c->[0]+$c->[1]*$t[$_]+$c->[2]*$t[$_]*$t[$_]))**2 for 0..$#u;
    ($x0+$tm,$c->[0]+$c->[1]*$tm+$c->[2]*$tm*$tm,sqrt(1.0/$c->[2]),sqrt($rr/@u)) }
# the three-node parabola (minimum and Delta chi2 = 1 half-width) through three (x,y)
sub parab3 { my ($n3)=@_; my ($x1,$x2,$x3)=map{$_->[0]}@$n3; my ($y1,$y2,$y3)=map{$_->[1]}@$n3;
    my $den=($x1-$x2)*($x1-$x3)*($x2-$x3);
    my $A=(($x3*($y2-$y1)+$x2*($y1-$y3)+$x1*($y3-$y2)))/$den;
    my $B=(($x3*$x3*($y1-$y2)+$x2*$x2*($y3-$y1)+$x1*$x1*($y2-$y3)))/$den;
    return (undef) unless $A>0;
    my $xm=-$B/(2*$A); ($xm,sqrt(1.0/$A),$y1-$A*($x1-$xm)**2) }
# fit a0 for one law node: parabola ladder in log10 a0 (converged totals); stores every total
sub fit_a0 { my ($nuf,$la_c,$law,$node,$x)=@_; $law//='law'; $node//='-';
    my $S=($EVS{$law}{$node}//={x=>$x,la=>{}}); my %v; my %keep;
    my $ev=sub{ my $la=shift; my $k=sprintf('%.6f',$la); return $v{$k} if defined $v{$k};
        my ($tot,$bY,$bq,$bB,$bc,$bn)=total_chi2(10**$la,$nuf); $v{$k}=$tot; $keep{$k}=[$bY,$bq,$bB,$bc,$bn];
        $S->{la}{$k}=[map{$bc->{$_}}@GNAMES]; $tot };
    $ev->($la_c+$_) for (-0.02,0,0.02);
    for (1..8) { my @k=sort{$a<=>$b} keys %v; my @r=parab_ls([@k],[map{$v{$_}}@k],15);
        my $nx; if (!@r || !defined $r[0]) { $nx= $v{$k[0]}<$v{$k[-1]} ? $k[0]-0.03 : $k[-1]+0.03 } else { $nx=$r[0] }
        $nx=$k[0]-0.04 if $nx<$k[0]-0.04; $nx=$k[-1]+0.04 if $nx>$k[-1]+0.04;
        my ($near)=sort{abs($a-$nx)<=>abs($b-$nx)}@k; last if abs($near-$nx)<0.003; $ev->($nx) }
    { my @k=sort{$a<=>$b} keys %v; my @r=parab_ls([@k],[map{$v{$_}}@k],10);
      if (defined $r[0]) { for my $d (-0.01,0.01) { my $t=$r[0]+$d; my ($near)=sort{abs($a-$t)<=>abs($b-$t)}@k; $ev->($t) if abs($near-$t)>0.004 } } }
    { my @k=sort{$a<=>$b} keys %v; my @r=parab_ls([@k],[map{$v{$_}}@k],10);
      if (defined $r[0]) { for my $d (-0.10,-0.05,0.05,0.10) { my $t=$r[0]+$d; my ($near)=sort{abs($a-$t)<=>abs($b-$t)}@k; $ev->($t) if abs($near-$t)>0.012 } } }
    my @k=sort{$a<=>$b} keys %v; my ($xm,$ym,$sx,$rms)=parab_ls([@k],[map{$v{$_}}@k],10);
    unless (defined $xm) { warn "  fit_a0 $law $node: no parabolic minimum, lowest node taken\n"; ($xm)=sort{$v{$a}<=>$v{$b}}@k; $xm+=0; $ym=$v{sprintf('%.6f',$xm)}; ($sx,$rms)=(0,0) }
    my $kb=sprintf('%.6f',$xm); $ev->($xm) unless defined $v{$kb};
    my ($bY,$bq,$bB,$bc,$bn)=@{$keep{$kb}};
    my %per=map{ $_=>[$bc->{$_},$bY->{$_},$bq->{$_},$bB->{$_},$bn->{$_}] } @GNAMES;
    my $a0=10**$xm;
    my @kk=sort{$a<=>$b} keys %v; my $nw=grep{$v{$_}<=$v{$kb}+10}@kk;
    $BESTFIT{$law}{$node}={a0=>$a0,la=>$xm,chi2=>$v{$kb},ymin=>$ym,sig_la=>$sx,sig_a0=>$a0*log(10.0)*$sx,rms=>$rms,n=>scalar(@kk),nwin=>$nw,lamin=>$kk[0],lamax=>$kk[-1],per=>\%per,x=>$x};
    ($a0,$v{$kb},$xm,$bY,$bq,$bB,$bc) }
# RAR scatter with marginalized nuisances: g_obs' = go/q^2, g_bar invariant
sub scatter_dex { my ($a0,$nuf,$bY,$bq,$bB)=@_; my @d; my $inv_a0=1.0/$a0;
    for my $g (@GNAMES){ my $Yd=$bY->{$g}; my $q2=$bq->{$g}**2; my $Yb=$bB?$bB->{$g}:$BRAT*$Yd;
        for my $p (@{$GAL{$g}}){ my $gbar=gbar_pt($g,$p,$Yd,$Yb); next if $gbar<=0;
            push @d, log10($p->[5]/$q2)-log10($nuf->($gbar*$inv_a0)*$gbar) }}
    my $n=@d; my $m=0; $m+=$_ for @d; $m/=$n; my $s=0; $s+=($_-$m)**2 for @d;
    (sqrt($s/$n),$m,$n) }
sub median { my @s=sort{$a<=>$b}@_; $s[int(@s/2)] }
sub cH0 { $C*$_[0]*1000/$MPC }

# --- the laws ---
my $mu_std=sub{my $X=$_[0]; $X/sqrt(1.0+$X*$X)};
my $mu_bcf=sub{my $X=$_[0]; $X*tanh(1.0/$X)};
my $mu_sqrt=sub{ my $xs=$_[0]; sub{my $X=$_[0]; my $b=$X*sqrt(1.0+($X/$xs)**2); $b/(1.0+$b)} };
my $nu_mcg=sub{ my $y=$_[0]; 1.0/(1.0-exp(-sqrt($y))) };
my %TABC;
sub law_nuf { my ($code,$c)=@_; my $k=defined $c ? "$code:".sprintf('%.6g',$c) : $code;
    return $TABC{$k} if $TABC{$k};
    my $f;
    if    ($code eq 'simple') { $f=\&nu_simple }
    elsif ($code eq 'T11') { $f=$nu_mcg }
    elsif ($code eq 'standard') { $f=tabf(build_table($mu_std)) }
    elsif ($code eq 'bcf') { $f=tabf(build_table($mu_bcf)) }
    elsif ($code eq 'sqrt-Xsat5') { $f=tabf(build_table($mu_sqrt->(5.0))) }
    elsif ($code eq 'sqrt-Xsat') { $f=tabf(build_table($mu_sqrt->($c))) }
    elsif ($code eq 'T3') { my $lam=log(1.0+$c); $f=tabf(build_table(sub{ mu_boost($_[0],$c,$lam) })) }
    else { die "no law $code\n" }
    $TABC{$k}=$f }

# ================= R. the reference fits (catalog distances and inclinations, same machinery) =================
my $T0=time;
my $REFLINE;
if ($SEC{R} && !$FROZEN) { local $FROZEN=1;
    my ($as,$cs)=fit_a0(law_nuf('simple'),log10(1.05e-10),'ref:simple','-');
    my (undef,$cq)=fit_a0(law_nuf('sqrt-Xsat5'),log10(1.04e-10),'ref:sqrt-Xsat5','-');
    my @nd; my $la=log10(1.0e-10);
    for my $d (0.10,0.15,0.20,0.25,0.30) { my ($a,$c,$l)=fit_a0(law_nuf('T3',$d),$la,'ref:T3',$d,$d); $la=$l; push @nd,[$d,$c] }
    my @s3=(sort{$a->[1]<=>$b->[1]}@nd)[0..2]; my ($dm)=parab3([sort{$a->[0]<=>$b->[0]}@s3]);
    $dm=sprintf('%.4f',$dm)+0; my (undef,$cd)=fit_a0(law_nuf('T3',$dm),$la,'ref:T3',"$dm",$dm);
    $REFLINE=sprintf("  simple a0=%.2e chi2=%.1f | sqrt-Xsat5 chi2=%.1f | T3 d=%.4f chi2=%.1f\n",$as,$cs,$cq,$dm,$cd);
    warn "  [reference line done, ".(time-$T0)." s]\n" }

# ================= A. fixed families, marginalized =================
my %fam;
$fam{$_}=law_nuf($_) for ('simple','standard','sqrt-Xsat5','T11','bcf');
# starting scales of the a0 ladders (converged fixed-bulge values; the ladder walks from there)
my %LA0=('simple'=>1.218e-10,'standard'=>1.470e-10,'sqrt-Xsat5'=>1.16e-10,'T11'=>1.178e-10,'bcf'=>1.52e-10);
if ($FROZEN) { %LA0=('simple'=>0.909e-10,'standard'=>1.166e-10,'sqrt-Xsat5'=>0.8985e-10,'T11'=>0.95e-10,'bcf'=>1.212e-10) }
my $SF = ($FREEB ? ($FROZEN?0.97:0.90) : 1.0) * ($PRIOR>0.15 ? 0.97 : 1.0);
$LA0{$_}*=$SF for keys %LA0;

print "R. reference fits with distances and inclinations at catalog values (same machinery and sample):\n";
print $REFLINE // "  (not computed in this run)\n"; print "\n";
printf "%-20s %11s %10s %9s %9s %8s %8s %8s %8s\n",
  'A. law','a0','chi2','scat(dex)','medYd','med f','med i-ic','D@67.4','D@73.0';
my %A0;
for my $name (!$SEC{A} ? () : ('simple','standard','sqrt-Xsat5','T11','bcf')) {
    my $nuf=$fam{$name};
    my ($a0,$chi,$la,$bY,$bq,$bB)=fit_a0($nuf,log10($LA0{$name}),$name,'-');
    my ($sc)=scatter_dex($a0,$nuf,$bY,$bq,$bB);
    # recover median implied f and i shift from the per-galaxy q via the split
    my (@fs,@dis);
    for my $g (@GNAMES){ my ($ic,$si,$sf)=@{$NUI{$g}}; my $q=$bq->{$g};
        my $sic=sin($ic*$D2R);
        my ($ilo,$ihi)=($ic-4.0*$si,$ic+4.0*$si);
        $ilo=10.0 if $ilo<10.0; $ihi=89.9 if $ihi>89.9;
        my ($ib,undef)=golden(sub{ my $i=$_[0]; my $f=($q*$sic/sin($i*$D2R))**2;
            (($f-1.0)/$sf)**2+(($i-$ic)/$si)**2 },$ilo,$ihi,22);
        push @fs,($q*$sic/sin($ib*$D2R))**2; push @dis,$ib-$ic; }
    @fs=sort{$a<=>$b}@fs; @dis=sort{$a<=>$b}@dis;
    my @ys=sort{$a<=>$b} map{$bY->{$_}}@GNAMES; my $medY=$ys[int(@ys/2)];
    my $D64=$C*67.4*1000/$MPC/$a0; my $D73=$C*73.0*1000/$MPC/$a0;
    $A0{$name}=$a0;
    printf "%-20s %11.4e %10.1f %9.4f %9.3f %8.3f %8.2f %8.3f %8.3f\n",
      $name,$a0,$chi,$sc,$medY,$fs[int(@fs/2)],$dis[int(@dis/2)],$D64,$D73;
    warn "  [$name done, ".(time-$T0)." s]\n";
}
if ($SEC{A}) {
    print "  (converged per-galaxy optimizer; a0 fitted by a parabola in log10 a0; chi2 evaluated at that a0;\n";
    printf "   bulge %s; disk prior %.2f dex%s; %s)\n", $FREEB?'mass-to-light free per galaxy':'fixed at 1.4 x disk',$PRIOR,
        $FREEB?sprintf(', bulge prior %.2f dex about %.2f',$PRIORB,$YB0):'', $FROZEN?'distances and inclinations held at catalog values':'distance and inclination marginalized';
    printf "  %-20s %10s %9s %10s %9s %9s %9s %9s\n",'law','a0 (1e-10)','+/- stat','chi2','D@67.4','D@70.0','D@73.0','medYb';
    for my $name ('simple','standard','sqrt-Xsat5','T11','bcf') { my $bt=$BESTFIT{$name}{'-'} or next;
        my @yb=sort{$a<=>$b} map{$bt->{per}{$_}[3]} grep{$BUL{$_}}@GNAMES;
        printf "  %-20s %10.4f %9.4f %10.3f %9.3f %9.3f %9.3f %9s\n",$name,$bt->{a0}*1e10,$bt->{sig_a0}*1e10,$bt->{chi2},
            cH0(67.4)/$bt->{a0},cH0(70.0)/$bt->{a0},cH0(73.0)/$bt->{a0},@yb?sprintf('%.3f',$yb[int(@yb/2)]):'-' }
}

# ================= B. the two-constant fit: the T3 scan over delta_fv =================
print "\n=== B. the boost law T3, ".($FROZEN?'catalog distances and inclinations':'marginalized').": chi^2 vs delta_fv, a0 refitted at each node (the two-constant fit) ===\n";
printf "  %-8s %8s %11s %10s %8s\n",'delta','lam','a0','chi2','D@67.4';
my @scan; my %DMIN;
sub t3_scan { my ($grid,$law,$la_start,$print)=@_; my @sc; my $la=$la_start;
    my %done;
    my $node=sub{ my $del=shift; my $k=sprintf('%.6g',$del); return $done{$k} if $done{$k};
        my $lam=log(1.0+$del); my $nuf=law_nuf('T3',$del);
        my ($a0,$chi,$l)=fit_a0($nuf,$la,$law,$k,$del); $la=$l; $done{$k}=[$del,$a0,$chi];
        printf "  %-8g %8.4f %11.4e %10.1f %8.3f\n",$del,$lam,$a0,$chi,$C*67.4*1000/$MPC/$a0 if $print;
        warn "  [$law delta=$del done, ".(time-$T0)." s]\n"; $done{$k} };
    push @sc,$node->($_) for @$grid;
    # extend upward (or downward) until the lowest node is interior
    for (1..6) { my @s=sort{$a->[0]<=>$b->[0]} values %done; my ($mn)=sort{$a->[2]<=>$b->[2]}@s;
        if ($mn==$s[-1]) { push @sc,$node->(sprintf('%.4f',$s[-1][0]+0.05)+0) } elsif ($mn==$s[0] && $s[0][0]>0.03) { push @sc,$node->(sprintf('%.4f',$s[0][0]*0.6)+0) } else { last } }
    [sort{$a->[0]<=>$b->[0]} values %done] }
my $LA_T3 = log10(($FROZEN?0.89e-10:1.17e-10)*$SF);
if ($SEC{B}) {
    my @g=@GRID; push @g,split(/,/,$OPT{extra_grid});
    my $res=t3_scan(\@g,'T3',$LA_T3,1); @scan=@$res;
}
# parabolic minimum + Dchi2=1 half-width from the three lowest nodes
my ($dmin,$sig,$ymin,$a0min,$chimin);
if (@scan>=3) {
my @s=sort{$a->[2]<=>$b->[2]} @scan; my @n3=sort{$a->[0]<=>$b->[0]} @s[0..2];
my ($x1,$x2,$x3)=map{$_->[0]}@n3; my ($y1,$y2,$y3)=map{$_->[2]}@n3;
my $den=($x1-$x2)*($x1-$x3)*($x2-$x3);
my $A=(($x3*($y2-$y1)+$x2*($y1-$y3)+$x1*($y3-$y2)))/$den;
my $B=(($x3*$x3*($y1-$y2)+$x2*$x2*($y3-$y1)+$x1*$x1*($y2-$y3)))/$den;
if ($A>0) { $dmin=-$B/(2*$A); $sig=sqrt(1.0/$A);
    printf "\n  parabolic: delta_fv = %.4f +/- %.4f (stat, Dchi2=1)\n",$dmin,$sig;
    my ($cor)=grep{abs($_->[0]-0.1725)<1e-6}@scan;
    $ymin=$y1-$A*($x1-$dmin)**2;   # parabola value at its minimum
    printf "  delta = 0.1725 (1/ln(1+delta) = 2 pi): Dchi2 = %+.1f vs parabolic min\n",
        $cor->[2]-$ymin if $cor;
    # a0 refitted at the parabolic minimum itself
    my ($nr)=sort{abs($a->[0]-$dmin)<=>abs($b->[0]-$dmin)}@scan;
    my $dk=sprintf('%.4f',$dmin)+0; ($a0min,$chimin)=fit_a0(law_nuf('T3',$dk),log10($nr->[1]),'T3*',"$dk",$dk);
    my $D64=$C*67.4*1000/$MPC/$a0min; my $D73=$C*73.0*1000/$MPC/$a0min;
    printf "  scale relation at the minimum: D\@67.4=%.3f D\@73=%.3f  product D ln(1+delta): %.3f / %.3f\n",
        $D64,$D73,$D64*log(1.0+$dmin),$D73*log(1.0+$dmin);
    print  "  node values: delta, a0 (1e-10 m/s^2) +/- stat, chi2 at the fitted a0, parabola minimum in log10 a0, its rms, points in the window / ladder\n";
    for my $nd (sort{$a->[0]<=>$b->[0]}@scan) { my $bfn=$BESTFIT{T3}{sprintf('%.6g',$nd->[0])} or next;
        printf "    %-8g %.5f +/- %.5f  %10.3f %10.3f %7.3f  %d/%d\n",$nd->[0],$bfn->{a0}*1e10,$bfn->{sig_a0}*1e10,$bfn->{chi2},$bfn->{ymin},$bfn->{rms},$bfn->{nwin},$bfn->{n} }
    { my @pm=map{ [$_->[0],$BESTFIT{T3}{sprintf('%.6g',$_->[0])}{ymin}] } @scan; my @s3=sort{$a->[0]<=>$b->[0]} (sort{$a->[1]<=>$b->[1]}@pm)[0..2];
      my ($xm,$sg)=parab3(\@s3); printf "  the same three-node rule on the parabola minima of the nodes: delta_fv = %.4f +/- %.4f\n",$xm,$sg if defined $xm }
    printf "  (minimum from the parabola through the three lowest nodes; a0 refitted at delta = %.4f:\n",$dk;
    printf "   a0 = %.4e +/- %.4e (stat), chi2 = %.2f there, parabola value %.2f; D\@70.0=%.3f, product %.3f)\n",$a0min,$BESTFIT{'T3*'}{"$dk"}{sig_a0},$chimin,$ymin,
        cH0(70.0)/$a0min,cH0(70.0)/$a0min*log(1.0+$dmin);
    $DMIN{x}=$dk;
} }

# ================= D. binned residuals of the simple function =================
print "\n=== D. binned mean residual (dex) in 0.5-dex bins of log10 y, MOND's simple function, ".($FROZEN?'catalog distances and inclinations':'marginalized')." ===\n";
if ($SEC{D} && $BESTFIT{'simple'}) { my $nuf=$fam{'simple'}; my $bt=$BESTFIT{'simple'}{'-'}; my $a0=$bt->{a0};
  my $inv_a0=1.0/$a0;
  my (@sum,@cnt);
  for my $g (@GNAMES){ my ($Yd,$q,$Yb)=@{$bt->{per}{$g}}[1,2,3]; my $q2=$q**2;
      for my $p (@{$GAL{$g}}){ my $gbar=gbar_pt($g,$p,$Yd,$Yb); next if $gbar<=0;
          my $ly=log10($gbar*$inv_a0);        # log10 of Y = gbar/a0
          my $bb=int(($ly+3.0)/0.5); next if $bb<0 || $bb>9;
          $sum[$bb]+=log10($p->[5]/$q2)-log10($nuf->($gbar*$inv_a0)*$gbar); $cnt[$bb]++ }}
  printf "  %-16s %6s %10s\n",'log10(gbar/a0)','N','mean(dex)';
  for my $bb (0..9){ next unless $cnt[$bb];
      printf "  [%+.1f,%+.1f) %8d %+10.4f\n",-3.0+0.5*$bb,-2.5+0.5*$bb,$cnt[$bb],$sum[$bb]/$cnt[$bb] }
}

# ================= D2. binned residuals of T3 at the minimum and of the standard function =================
if ($SEC{D} && $DMIN{x} && $BESTFIT{standard}) {
    print "\n=== D2. binned mean residual (dex) in 0.25-dex bins of log10 g_bar: T3 at its two-constant minimum and MOND's standard function ===\n";
    my @set=(['T3',law_nuf('T3',$DMIN{x}),$BESTFIT{'T3*'}{$DMIN{x}}],['standard',$fam{standard},$BESTFIT{standard}{'-'}]);
    my (%sum,%cnt);
    for my $st (@set) { my ($lab,$nuf,$bf)=@$st; my $ia=1.0/$bf->{a0};
        for my $g (@GNAMES) { my ($Yd,$q,$Yb)=@{$bf->{per}{$g}}[1,2,3]; my $q2=$q*$q;
            for my $p (@{$GAL{$g}}) { my $gb=gbar_pt($g,$p,$Yd,$Yb); next if $gb<=0; my $bb=POSIX::floor((log10($gb)+12.6)/0.25);
                $sum{$lab}{$bb}+=log10($p->[5]/$q2)-log10($nuf->($gb*$ia)*$gb); $cnt{$lab}{$bb}++ } } }
    printf "  %-16s %8s %10s %8s %10s\n",'log10 g_bar','N T3','T3 (dex)','N std','std (dex)';
    my ($mx,$mxb)=(0,'');
    for my $bb (sort{$a<=>$b} keys %{$cnt{T3}}) { my ($nc,$ns)=($cnt{T3}{$bb}//0,$cnt{standard}{$bb}//0); next unless $nc>=8 && $ns>=8;
        my ($rc,$rs)=($sum{T3}{$bb}/$nc,$sum{standard}{$bb}/$ns); my $lo=-12.6+0.25*$bb;
        printf "  [%6.2f,%6.2f) %8d %+10.4f %8d %+10.4f\n",$lo,$lo+0.25,$nc,$rc,$ns,$rs; if (abs($rc)>abs($mx)) { $mx=$rc; $mxb=sprintf('%.2f',$lo) } }
    printf "  largest |T3 residual| over bins both populate (>= 8 points): %.4f dex (bin from %s)\n",abs($mx),$mxb;
}

# ================= E. the square-root form with X_sat fitted (sqrt-Xsat) =================
my %ONS;
if ($SEC{E}) {
    print "\n=== E. the square-root form, mu = b/(1+b) with b = X sqrt(1+(X/X_sat)^2), ".($FROZEN?'catalog distances and inclinations':'marginalized').": chi^2 vs X_sat, a0 refitted at each node ===\n";
    printf "  %-8s %11s %10s %8s\n",'X_sat','a0','chi2','D@67.4';
    my %done; my $la=log10(($FROZEN?0.90e-10:1.15e-10)*$SF);
    my $node=sub{ my $xs=shift; my $k=sprintf('%.6g',$xs); return $done{$k} if $done{$k};
        my ($a0,$chi,$l)=fit_a0(law_nuf('sqrt-Xsat',$xs),$la,'sqrt-Xsat',$k,$xs); $la=$l; $done{$k}=[$xs,$a0,$chi];
        printf "  %-8g %11.4e %10.1f %8.3f\n",$xs,$a0,$chi,cH0(67.4)/$a0; warn "  [sqrt-Xsat X_sat=$xs done, ".(time-$T0)." s]\n"; $done{$k} };
    $node->($_) for split /,/,$OPT{onset_grid};
    for (1..6) { my @s=sort{$a->[0]<=>$b->[0]} values %done; my ($mn)=sort{$a->[2]<=>$b->[2]}@s;
        if ($mn==$s[-1]) { $node->($s[-1][0]+1.0) } elsif ($mn==$s[0] && $s[0][0]>0.6) { $node->($s[0][0]-0.5) } else { last } }
    my @s=sort{$a->[0]<=>$b->[0]} values %done; my @n3=sort{$a->[0]<=>$b->[0]} (sort{$a->[2]<=>$b->[2]}@s)[0..2];
    my ($xm,$sg,$ym)=parab3([map{[$_->[0],$_->[2]]}@n3]);
    print  "  node values: X_sat, a0 (1e-10 m/s^2) +/- stat, chi2 at the fitted a0, parabola minimum in log10 a0\n";
    for my $nd (@s) { my $bfn=$BESTFIT{'sqrt-Xsat'}{sprintf('%.6g',$nd->[0])} or next;
        printf "    %-8g %.5f +/- %.5f  %10.3f %10.3f\n",$nd->[0],$bfn->{a0}*1e10,$bfn->{sig_a0}*1e10,$bfn->{chi2},$bfn->{ymin} }
    if (defined $xm) { printf "\n  parabolic: X_sat = %.4f +/- %.4f (stat, Dchi2=1), parabola value %.2f\n",$xm,$sg,$ym;
        my $xk=sprintf('%.4f',$xm)+0; my ($a0x,$chx)=fit_a0(law_nuf('sqrt-Xsat',$xk),$la,'sqrt-Xsat*',"$xk",$xk);
        printf "  refitted at X_sat = %.4f: a0 = %.4e +/- %.4e (stat), chi2 = %.2f\n",$xk,$a0x,$BESTFIT{'sqrt-Xsat*'}{"$xk"}{sig_a0},$chx;
        %ONS=(x=>$xk,sig=>$sg,a0=>$a0x,chi2=>$chx,ymin=>$ym) }
    else { print "  (no interior parabolic minimum)\n" }
}

# ================= the clustered bootstrap over the stored profiles =================
my $seed=$OPT{seed};
sub rng { $seed=($seed*1103515245+12345)%2147483648; $seed/2147483648 }
my @IDX=0..$#GNAMES;
# minimum of one law node for given galaxy weights: parabola in log10 a0 through the stored totals
my %EXTRAP=(n=>0,out=>0);
sub node_min { my ($law,$node,$w)=@_; my $S=$EVS{$law}{$node}; my @k=sort{$a<=>$b} keys %{$S->{la}};
    my @y=map{ my $v=$S->{la}{$_}; my $t=0; if ($w) { $t+=$w->[$_]*$v->[$_] for @IDX } else { $t+=$_ for @$v } $t } @k;
    my ($xm,$ym)=parab_ls([@k],\@y,10);
    if ($w) { $EXTRAP{n}++; if (!defined $xm || $xm<$k[0] || $xm>$k[-1]) { $EXTRAP{out}++;
        my ($i)=sort{$y[$a]<=>$y[$b]}0..$#y; return ($y[$i],$k[$i]) } }     # draws: the lowest node at the end of the ladder
    return ($ym,$xm) if defined $ym;
    my ($i)=sort{$y[$a]<=>$y[$b]}0..$#y; ($y[$i],$k[$i]) }
# minimum of a scanned law: the three-lowest-node parabola over the grid nodes
my %SEDGE=(n=>0,edge=>0);
sub scan_min { my ($law,$w)=@_; my @n=sort{$a->[0]<=>$b->[0]} map{ my $S=$EVS{$law}{$_}; my ($ym,$la)=node_min($law,$_,$w); [$S->{x},$ym,$la] } keys %{$EVS{$law}};
    return undef if @n<3; my @n3=sort{$a->[0]<=>$b->[0]} (sort{$a->[1]<=>$b->[1]}@n)[0..2];
    my ($xm,$sg,$ym)=parab3([map{[$_->[0],$_->[1]]}@n3]);
    if ($w) { $SEDGE{n}++; my ($ib)=sort{$n[$a][1]<=>$n[$b][1]}0..$#n;
        if ($ib==0 || $ib==$#n) { $SEDGE{edge}++; return ($n[$ib][1],$n[$ib][0]) }      # lowest node on the edge of the grid: no extrapolation
        if (!defined $xm || $xm<$n3[0][0] || $xm>$n3[2][0]) { ($xm,$sg,$ym)=parab3([map{[$_->[0],$_->[1]]}@n[$ib-1..$ib+1]]);
            return ($n[$ib][1],$n[$ib][0]) if !defined $xm || $xm<$n[$ib-1][0] || $xm>$n[$ib+1][0] } }
    return ($n3[1][1],$n3[1][0]) unless defined $xm; ($ym,$xm) }
# the same minimum, with log10 a0 at the minimum interpolated through the per-node a0 minima of the three nodes
sub scan_min_a0 { my ($law,$w)=@_; my @n=sort{$a->[0]<=>$b->[0]} map{ my $S=$EVS{$law}{$_}; my ($ym,$la)=node_min($law,$_,$w); [$S->{x},$ym,$la] } keys %{$EVS{$law}};
    return () if @n<3; my ($ib)=sort{$n[$a][1]<=>$n[$b][1]}0..$#n;
    return ($n[$ib][1],$n[$ib][0],$n[$ib][2]) if $ib==0 || $ib==$#n;
    my @n3=sort{$a->[0]<=>$b->[0]} (sort{$a->[1]<=>$b->[1]}@n)[0..2];
    my ($xm,$sg,$ym)=parab3([map{[$_->[0],$_->[1]]}@n3]);
    if (!defined $xm || $xm<$n3[0][0] || $xm>$n3[2][0]) { @n3=@n[$ib-1..$ib+1]; ($xm,$sg,$ym)=parab3([map{[$_->[0],$_->[1]]}@n3]);
        return ($n[$ib][1],$n[$ib][0],$n[$ib][2]) if !defined $xm || $xm<$n3[0][0] || $xm>$n3[2][0] }
    my ($x1,$x2,$x3)=map{$_->[0]}@n3; my ($l1,$l2,$l3)=map{$_->[2]}@n3;      # Lagrange interpolation of log10 a0 in delta
    my $la=$l1*($xm-$x2)*($xm-$x3)/(($x1-$x2)*($x1-$x3))+$l2*($xm-$x1)*($xm-$x3)/(($x2-$x1)*($x2-$x3))+$l3*($xm-$x1)*($xm-$x2)/(($x3-$x1)*($x3-$x2));
    ($ym,$xm,$la) }
my (@KB,%KBC);   # per-draw product at H0 = 67.4 and 73 for T3 (section M)
sub law_min { my ($spec,$w)=@_; my ($law,$node)=@$spec; return scan_min($law,$w) if $node eq '*'; node_min($law,$node,$w) }
# delta-method clustered error of a difference of two best-fit evaluations
sub dmeth { my ($A,$B,$sub)=@_; my @gs=grep{ !$sub || $sub->($_) } @GNAMES; my @d=map{ $A->{per}{$_}[0]-$B->{per}{$_}[0] } @gs;
    my $n=@d; return (0,0,$n) if $n<1; my $s=0; $s+=$_ for @d; return ($s,0,$n) if $n<2; my $m=$s/$n; my $v=0; $v+=($_-$m)**2 for @d; ($s,sqrt($v*$n/($n-1)),$n) }
my %CL;  # clustered results: name => [central, boot sd, delta-method value, delta se, bulge, bulgeless...]
if ($SEC{F}) {
    my $dk=$DMIN{x}; my $xk=$ONS{x};
    my @M=(['standard - simple',['standard','-'],['simple','-']],
           ['bcf (sharp knee) - simple',['bcf','-'],['simple','-']],
           ['simple - T3',['simple','-'],['T3','*']],
           ['standard - T3',['standard','-'],['T3','*']],
           ['T11 - T3',['T11','-'],['T3','*']],
           ['sqrt-Xsat5 - T3',['sqrt-Xsat5','-'],['T3','*']],
           ['sqrt-Xsat5 - T3 at delta = 0.20',['sqrt-Xsat5','-'],['T3','0.2']],
           ['sqrt-Xsat - T3',['sqrt-Xsat','*'],['T3','*']],
           ['T3 at delta = 0.1725 - T3',['T3','0.1725'],['T3','*']],
           ['simple - T11',['simple','-'],['T11','-']]);
    @M=grep{ my $ok=1; for my $sp (@{$_}[1,2]) { my ($l,$n)=@$sp; $ok=0 unless $EVS{$l} && ($n eq '*' ? scalar(keys %{$EVS{$l}})>=3 : $EVS{$l}{$n}) } $ok } @M;
    my %BF=( 'T3,*'=> ($dk && $BESTFIT{'T3*'}{"$dk"}), 'sqrt-Xsat,*'=>($xk && $BESTFIT{'sqrt-Xsat*'}{"$xk"}) );
    my $bf=sub{ my ($l,$n)=@{$_[0]}; $n eq '*' ? $BF{"$l,*"} : $BESTFIT{$l}{$n} };
    my (%s1,%s2,%pd,@dd,@xx); my $NB=$OPT{nboot}; my %ps1; my %ps2;
    my %cen; for my $m (@M) { my ($ya)=law_min($m->[1]); my ($yb)=law_min($m->[2]); $cen{$m->[0]}=$ya-$yb }
    my @PAR=(['delta_fv (T3)','T3'],['X_sat (onset)','sqrt-Xsat']);
    @PAR=grep{ $EVS{$_->[1]} && scalar(keys %{$EVS{$_->[1]}})>=3 } @PAR;
    my %pa1; my %pa2; my %pn;
    for my $ib (1..$NB) { my @w=(0)x@GNAMES; $w[int(rng()*@GNAMES)]++ for @IDX;
        my %cache; my $lm=sub{ my $sp=shift; my $k=join(',',@$sp); $cache{$k}//=[law_min($sp,\@w)]; @{$cache{$k}} };
        for my $m (@M) { my ($ya)=$lm->($m->[1]); my ($yb)=$lm->($m->[2]); my $d=$ya-$yb; $s1{$m->[0]}+=$d; $s2{$m->[0]}+=$d*$d }
        for my $m (@M) { my ($ya)=$lm->($m->[1]); my ($yb)=$lm->($m->[2]); push @{$pd{$m->[0]}},$ya-$yb }
        for my $p (@PAR) { my (undef,$x)=$lm->([$p->[1],'*']); next unless defined $x; $pa1{$p->[0]}+=$x; $pa2{$p->[0]}+=$x*$x; $pn{$p->[0]}++; push @{$pd{$p->[0]}},$x }
        for my $l ('T11','simple') { next unless $EVS{$l}; my (undef,$la)=$lm->([$l,'-']); $pa1{"a0 $l"}+=10**$la; $pa2{"a0 $l"}+=10**(2*$la); $pn{"a0 $l"}++; push @{$pd{"a0 $l"}},10**$la }
        if ($EVS{T3} && scalar(keys %{$EVS{T3}})>=3) { my (undef,$xd,$ld)=scan_min_a0('T3',\@w);
            push @KB,[$xd,10**$ld,cH0(67.4)/10**$ld*log(1+$xd),cH0(73.0)/10**$ld*log(1+$xd)] if defined $ld }
    }
    if ($EVS{T3} && scalar(keys %{$EVS{T3}})>=3) { my (undef,$xd,$ld)=scan_min_a0('T3',undef); %KBC=(d=>$xd,a0=>10**$ld,k67=>cH0(67.4)/10**$ld*log(1+$xd),k73=>cH0(73.0)/10**$ld*log(1+$xd)) }
    my $pct=sub{ my @x=sort{$a<=>$b}@{$_[0]}; my $n=@x; return (0,0,0) unless $n; ($x[int(0.16*$n)],$x[int(0.5*$n)],$x[int(0.84*$n)]) };
    print "\n=== F. Delta chi2 margins, galaxy-clustered ===\n";
    printf "  %-38s %10s %9s %7s %9s   %10s %9s   %s\n",'margin','dchi2','+/-boot','sigma','16-84/2','best fits','+/-delta','bulge / bulgeless (delta method)';
    for my $m (@M) { my $n=$m->[0]; my $mb=$s1{$n}/$NB; my $sd=sqrt($s2{$n}/$NB-$mb*$mb);
        my ($A,$B)=($bf->($m->[1]),$bf->($m->[2])); my ($dv,$se,$sb,$seb,$sn,$sen)=('-','-','','','','');
        if ($A && $B) { ($dv,$se)=dmeth($A,$B); ($sb,$seb)=dmeth($A,$B,sub{$BUL{$_[0]}}); ($sn,$sen)=dmeth($A,$B,sub{!$BUL{$_[0]}}) }
        $CL{$n}=[$cen{$n},$sd,$dv,$se]; my ($p16,undef,$p84)=$pct->($pd{$n});
        printf "  %-38s %+10.2f %9.2f %7.2f %9.2f   %s %s   %s\n",$n,$cen{$n},$sd,$sd>0?abs($cen{$n})/$sd:0,($p84-$p16)/2,
            ($dv eq '-'?sprintf('%10s','-'):sprintf('%+10.2f',$dv)),($se eq '-'?sprintf('%9s','-'):sprintf('%9.2f',$se)),
            ($A&&$B)?sprintf('%+.2f +/- %.2f / %+.2f +/- %.2f',$sb,$seb,$sn,$sen):'-' }
    printf "  (dchi2: profile minima of the full sample; +/-boot: SD over %d galaxy-bootstrap draws, each law refitted along its\n",$NB;
    print  "   stored profile, a parabola in log10 a0 per node and the three-lowest-node parabola across shape nodes;\n";
    print  "   best fits: per-galaxy difference at the evaluated best fits; +/-delta: delta-method clustered error)\n";
    printf "  (in the draws, node minima beyond the ends of their a0 ladder, taken at the ladder's lowest node: %d of %d; shape minima on the edge of the grid: %d of %d)\n",$EXTRAP{out},$EXTRAP{n},$SEDGE{edge},$SEDGE{n};
    print  "  clustered errors of the fitted constants (bootstrap SD along the stored profiles):\n";
    for my $p (sort keys %pn) { my $m=$pa1{$p}/$pn{$p}; my $sd=sqrt($pa2{$p}/$pn{$p}-$m*$m); my ($p16,$p50,$p84)=$pct->($pd{$p}); my $sc=($p=~/^a0/)?1e10:1;
        printf "    %-26s mean %.4f  SD %.4f  median %.4f  16-84 percent %.4f to %.4f (half-width %.4f)%s  (%d draws)\n",$p,$m*$sc,$sd*$sc,$p50*$sc,$p16*$sc,$p84*$sc,($p84-$p16)/2*$sc,($p=~/^a0/?' e-10':''),$pn{$p} }
    $CL{'_dpct'}=[$pct->($pd{'delta_fv (T3)'})] if $pd{'delta_fv (T3)'};
    $CL{'_a0pct_T11'}=[$pct->($pd{'a0 T11'})] if $pd{'a0 T11'};
    if ($pd{'delta_fv (T3)'}) { my @x=@{$pd{'delta_fv (T3)'}}; my $nb=grep{$_<=0.1725}@x;
        printf "    delta = 0.1725: %d of %d draws put the T3 minimum at or below it\n",$nb,scalar(@x) }
    # delta-method (sandwich) clustered errors of the constants, from the per-galaxy values stored along the profiles
    print  "  delta-method (sandwich) clustered errors, from per-galaxy slopes along the stored profiles:\n";
    for my $l ('T11','simple') { next unless $EVS{$l} && $BESTFIT{$l}{'-'}; my $bt=$BESTFIT{$l}{'-'}; my $S=$EVS{$l}{'-'}{la};
        my @k=sort{$a<=>$b} grep{ abs($_-$bt->{la})<0.026 } keys %$S; next unless @k>=3;
        my ($Cs,@sg)=(0); for my $gi (@IDX) { my $c=lsq([map{[1,$_-$bt->{la},($_-$bt->{la})**2]}@k],[map{$S->{$_}[$gi]}@k]); push @sg,$c->[1]; $Cs+=2*$c->[2] }
        my $m=0; $m+=$_ for @sg; $m/=@sg; my $v=0; $v+=($_-$m)**2 for @sg; $v*=@sg/(@sg-1); my $sla=sqrt($v)/$Cs;
        printf "    a0 %-20s +/- %.4f e-10 (stat %.4f; ratio %.2f)\n",$l,$bt->{a0}*log(10)*$sla*1e10,$bt->{sig_a0}*1e10,$sla/$bt->{sig_la};
        $CL{"_a0dm_$l"}=$bt->{a0}*log(10)*$sla }
    for my $P (['delta_fv (T3)','T3'],['X_sat (onset)','sqrt-Xsat']) { my ($nm,$l)=@$P; next unless $EVS{$l};
        my @nd=sort{$a->[0]<=>$b->[0]} map{ [$EVS{$l}{$_}{x},$BESTFIT{$l}{$_}] } grep{ $BESTFIT{$l}{$_} } keys %{$EVS{$l}}; next if @nd<3;
        my @n3=sort{$a->[0]<=>$b->[0]} (sort{$a->[1]{chi2}<=>$b->[1]{chi2}}@nd)[0..2];
        my ($xm,$sx)=parab3([map{[$_->[0],$_->[1]{chi2}]}@n3]); next unless defined $xm;
        my ($Cs,@sg)=(0); for my $g (@GNAMES) { my ($x1,$x2,$x3)=map{$_->[0]}@n3; my ($y1,$y2,$y3)=map{$_->[1]{per}{$g}[0]}@n3;
            my $den=($x1-$x2)*($x1-$x3)*($x2-$x3); my $A=(($x3*($y2-$y1)+$x2*($y1-$y3)+$x1*($y3-$y2)))/$den; my $B=(($x3*$x3*($y1-$y2)+$x2*$x2*($y3-$y1)+$x1*$x1*($y2-$y3)))/$den;
            push @sg,2*$A*$xm+$B; $Cs+=2*$A }
        my $m=0; $m+=$_ for @sg; $m/=@sg; my $v=0; $v+=($_-$m)**2 for @sg; $v*=@sg/(@sg-1);
        printf "    %-23s +/- %.4f (stat %.4f; ratio %.2f), per-galaxy profiles through the nodes %s\n",$nm,sqrt($v)/$Cs,$sx,sqrt($v)/$Cs/$sx,join(' ',map{$_->[0]}@n3);
        $CL{"_dm_$l"}=sqrt($v)/$Cs }
    $CL{'_a0sd_T11'}= $pn{'a0 T11'} ? sqrt($pa2{'a0 T11'}/$pn{'a0 T11'}-($pa1{'a0 T11'}/$pn{'a0 T11'})**2) : undef;
    $CL{'_dsd'}= $pn{'delta_fv (T3)'} ? sqrt($pa2{'delta_fv (T3)'}/$pn{'delta_fv (T3)'}-($pa1{'delta_fv (T3)'}/$pn{'delta_fv (T3)'})**2) : undef;
    $CL{'_xsd'}= $pn{'X_sat (onset)'} ? sqrt($pa2{'X_sat (onset)'}/$pn{'X_sat (onset)'}-($pa1{'X_sat (onset)'}/$pn{'X_sat (onset)'})**2) : undef;
    # where sqrt-Xsat gains on T3: the galaxies reaching y > 8 under T3's fit, and the largest single gains
    if ($BF{'sqrt-Xsat,*'} && $BF{'T3,*'}) { my ($A,$B)=($BF{'sqrt-Xsat,*'},$BF{'T3,*'}); my $a0c=$B->{a0}; my @hy;
        for my $g (@GNAMES) { my ($Yd,$q,$Yb)=@{$B->{per}{$g}}[1,2,3]; my $hit=0; for my $p (@{$GAL{$g}}) { my $gb=gbar_pt($g,$p,$Yd,$Yb); $hit=1 if $gb>0 && $gb/$a0c>8 } push @hy,$g if $hit }
        my ($tot)=dmeth($A,$B); my ($th)=dmeth($A,$B,sub{ my $g=$_[0]; scalar grep{$_ eq $g}@hy });
        my @d=sort{$a->[1]<=>$b->[1]} map{[$_,$A->{per}{$_}[0]-$B->{per}{$_}[0]]}@GNAMES;
        printf "  sqrt-Xsat - T3 at the best fits: %+.2f in all, %+.2f in the %d galaxies reaching y > 8 under T3's fit;\n",$tot,$th,scalar(@hy);
        printf "    largest single galaxies: %s\n",join(', ',map{sprintf('%s %+.2f',@$_)}@d[0..4]) }
}

# ================= G. the fitted scales and D = cH0/a0 =================
if ($SEC{G} && $BESTFIT{T11}) {
    print "\n=== G. the fitted scales and D = cH0/a0 ===\n";
    my $bt=$BESTFIT{T11}{'-'}; my $at=$bt->{a0}; my $s=$bt->{sig_a0}; my $sc=$CL{'_a0sd_T11'};
    printf "  T11 (the empirical function) a0 = %.4f +/- %.4f e-10 (stat, Dchi2=1, nuisances re-optimized)%s; chi2 %.2f\n",$at*1e10,$s*1e10,
        defined $sc?sprintf('; galaxy-clustered +/- %.4f',$sc*1e10):'',$bt->{chi2};
    for my $H (67.4,70.0,73.0) { my $a6=cH0($H)/6; printf "  H0 = %.1f: D = cH0/a0 = %.3f; a0 at D = 6 is %.4f e-10, %+.2f sigma (stat)%s\n",$H,cH0($H)/$at,$a6*1e10,($at-$a6)/$s,
        defined $sc?sprintf(', %+.2f sigma (clustered)',($at-$a6)/$sc):'' }
    print  "  (D = cH0/a0 at H0 itself, and Paper I's divisor cH_dS/a0 is 0.8275 D; sigma: a0 minus the a0 at D = 6, over its stat or clustered error)\n";
    for my $l ('simple','sqrt-Xsat5','standard') { my $bb=$BESTFIT{$l}{'-'} or next;
        printf "  %-20s a0 = %.4f +/- %.4f e-10: D = %.3f (67.4), %.3f (70.0), %.3f (73.0)\n",$l,$bb->{a0}*1e10,$bb->{sig_a0}*1e10,map{cH0($_)/$bb->{a0}}(67.4,70.0,73.0) }
    if ($a0min) { printf "  T3 at its two-constant minimum a0 = %.4f e-10: D = %.3f (67.4), %.3f (70.0), %.3f (73.0)\n",$a0min*1e10,map{cH0($_)/$a0min}(67.4,70.0,73.0) }
    { my @f=map{ [$_, cH0($_)/(2*$PI)] } (67.4,73.0);
      printf "  a0 = cH0/2pi: %.4f e-10 (67.4), %.4f e-10 (73.0); T11 against it %+.1f%% to %+.1f%%%s\n",$f[0][1]*1e10,$f[1][1]*1e10,
          100*($at/$f[0][1]-1),100*($at/$f[1][1]-1), $a0min?sprintf(', T3 at its minimum %+.1f%% to %+.1f%%',100*($a0min/$f[0][1]-1),100*($a0min/$f[1][1]-1)):'' }
}

# ================= H. the product and the quantities derived from delta_fv =================
if ($SEC{H} && defined $dmin) {
    print "\n=== H. the product D ln(1+delta) and the quantities derived from delta_fv at the two-constant minimum ===\n";
    my $lam=log(1.0+$dmin); my $lp=log(1.0+$dmin+$sig); my $lmn=log(1.0+$dmin-$sig);
    # a0 along the profile: quadratic in delta through the nodes nearest the minimum
    my @n=sort{abs($a->[0]-$dmin)<=>abs($b->[0]-$dmin)} @scan; my @u=sort{$a->[0]<=>$b->[0]} @n[0..($#n<4?$#n:4)];
    my $ca=lsq([map{[1,$_->[0],$_->[0]**2]}@u],[map{log10($_->[1])}@u]);
    my $a0d=sub{ my $x=$_[0]; 10**($ca->[0]+$ca->[1]*$x+$ca->[2]*$x*$x) };
    for my $H (67.4,73.0) { my $k=sub{ cH0($H)/$a0d->($_[0])*log(1.0+$_[0]) }; my $k0=cH0($H)/$a0min*$lam;
        my ($km,$kp)=($k->($dmin-$sig),$k->($dmin+$sig)); my $sk=($kp-$km)/2;
        printf "  H0 = %.1f: product = %.3f at the minimum (a0 refitted there); over delta +/- sigma %.3f to %.3f; unity %.2f sigma away\n",$H,$k0,$km,$kp,abs($k0-1)/$sk }
    my ($cor)=grep{abs($_->[0]-0.1725)<1e-6}@scan;
    if ($cor) { my $dc=$cor->[2]-$ymin; printf "  delta = 0.1725 (1/ln(1+delta) = 2 pi): Dchi2 = %+.2f against the parabolic minimum, %.2f sigma for one constant%s\n",$dc,sqrt($dc>0?$dc:0),
        $CL{'T3 at delta = 0.1725 - T3'}?sprintf('; profile margin %+.2f +/- %.2f (clustered)',@{$CL{'T3 at delta = 0.1725 - T3'}}[0,1]):'' }
    my $xi=0.6573; { my ($lo,$hi)=(0.3,1.0); for (1..80) { my $m=0.5*($lo+$hi); if (exp($m)-1-sqrt(2.0)*$m<0) {$lo=$m} else {$hi=$m} } $xi=0.5*($lo+$hi) }
    printf "  derived from delta_fv = %.4f +/- %.4f (stat): lambda = ln(1+delta) = %.4f\n",$dmin,$sig,$lam;
    printf "    hardening onset X_onset = xi*/lambda = %.3f (xi* = %.4f, root of e^u - 1 = sqrt2 u); over delta +/- sigma %.2f to %.2f\n",$xi/$lam,$xi,$xi/$lp,$xi/$lmn;
    printf "    derived divisor D = 1/lambda = %.3f, +/- %.3f from the stat error (%.3f to %.3f)\n",1/$lam,$sig/((1+$dmin)*$lam*$lam),1/$lp,1/$lmn;
    printf "    the counting factor kappa = 1/3 would give D = 1/(3 lambda) = %.3f\n",1/(3*$lam);
    printf "    the scale derived at kappa = 1, a0 = cH0 lambda: %.4f e-10 (67.4), %.4f e-10 (73.0), against the fitted %.4f e-10 at the minimum\n",cH0(67.4)*$lam*1e10,cH0(73.0)*$lam*1e10,$a0min*1e10;
    printf "    the amplification of Paper II Eq. (4), A = 2/(Z1 delta^2) at Z1 = 1: %.1f (%.1f to %.1f over delta -/+ sigma)\n",2/$dmin**2,2/($dmin+$sig)**2,2/($dmin-$sig)**2;
    { my $k67=cH0(67.4)/$a0min*$lam; my $kp=cH0(67.4)/$a0d->($dmin+$sig)*log(1+$dmin+$sig); my $km=cH0(67.4)/$a0d->($dmin-$sig)*log(1+$dmin-$sig);
      my $rel=($kp-$km)/2/$k67; printf "    1/A = delta^2/2 = %.4f, against the relative stat precision %.3f of the product at H0 = 67.4, %.1f times larger\n",$dmin**2/2,$rel,$rel/($dmin**2/2) }
    # the product under the square-root form: the lambda of T3 whose onset xi*/lambda sits at the fitted X_sat
    if (%ONS) { my $lo=$xi/$ONS{x}; my $l5=$xi/5.0; my $b5=$BESTFIT{'sqrt-Xsat5'}{'-'};
        printf "  the product under the square-root form (D at its a0 times xi*/X_sat): X_sat = %.3f, a0 %.4f e-10: %.3f (67.4), %.3f (73.0)\n",
            $ONS{x},$ONS{a0}*1e10,cH0(67.4)/$ONS{a0}*$lo,cH0(73.0)/$ONS{a0}*$lo;
        printf "    the same at the fixed X_sat = 5 of sqrt-Xsat5 (a0 %.4f e-10): %.3f (67.4), %.3f (73.0)\n",$b5->{a0}*1e10,cH0(67.4)/$b5->{a0}*$l5,cH0(73.0)/$b5->{a0}*$l5 if $b5 }
    # T3 at delta_fv against the empirical function (T11) over 0.01 <= y <= 32, both at the same a0
    my $nuc=law_nuf('T3',sprintf('%.4f',$dmin)+0); my ($mx,$ss,$nn)=(0,0,0);
    for (my $ly=-2;$ly<=log10(32)+1e-9;$ly+=0.01) { my $y=10**$ly; my $d=log10($nuc->($y))-log10($nu_mcg->($y)); $mx=abs($d) if abs($d)>$mx; $ss+=$d*$d; $nn++ }
    printf "  T3 at delta = %.4f against the empirical function (T11) over 0.01 <= y <= 32: rms %.4f dex, largest %.4f dex\n",$dmin,sqrt($ss/$nn),$mx;
}

# ================= I. the high-acceleration bands and the inclination correlation (diagnostics) =================
if ($SEC{I} && $DMIN{x}) {
    print "\n=== I. the bands 4 <= y <= 8 and y > 8 at T3's two-constant fit (diagnostics only, not in chi2) ===\n";
    my $ctl=$BESTFIT{'T3*'}{$DMIN{x}}; my $a0c=$ctl->{a0};
    my %LAWS=('T3'=>[law_nuf('T3',$DMIN{x}),$ctl],'T11'=>[$fam{T11},$BESTFIT{T11}{'-'}],
              'simple'=>[$fam{'simple'},$BESTFIT{'simple'}{'-'}],'standard'=>[$fam{standard},$BESTFIT{standard}{'-'}],
              'sqrt-Xsat5'=>[$fam{'sqrt-Xsat5'},$BESTFIT{'sqrt-Xsat5'}{'-'}]);
    $LAWS{'sqrt-Xsat'}=[law_nuf('sqrt-Xsat',$ONS{x}),$BESTFIT{'sqrt-Xsat*'}{$ONS{x}}] if %ONS;
    for my $band ([4,8],[8,1e9]) { my ($ylo,$yhi)=@$band; my @mem;
        for my $g (@GNAMES) { my ($Yd,$q,$Yb)=@{$ctl->{per}{$g}}[1,2,3]; my $i=0;
            for my $p (@{$GAL{$g}}) { my $gb=gbar_pt($g,$p,$Yd,$Yb); my $y=$gb/$a0c; push @mem,[$g,$i] if $gb>0 && $y>=$ylo && $y<=$yhi; $i++ } }
        my %ng; $ng{$_->[0]}++ for @mem; next unless @mem>=2;
        my $stat=sub{ my ($nuf,$fit,$own)=@_; my $src=$own?$fit:$ctl; my (%so,%sp,%sr,%cnt);
            for my $m (@mem) { my ($g,$i)=@$m; my ($Yd,$q,$Yb)=@{$src->{per}{$g}}[1,2,3]; my $p=$GAL{$g}[$i];
                my $gb=gbar_pt($g,$p,$Yd,$Yb); next if $gb<=0; my $obs=log10($p->[5]/($q*$q))-log10($gb); my $pr=log10($nuf->($gb/$fit->{a0}));
                $so{$g}+=$obs; $sp{$g}+=$pr; $sr{$g}+=$obs-$pr; $cnt{$g}++ }
            my @gg=sort keys %cnt; my $N=0; $N+=$cnt{$_} for @gg; my ($O,$P,$R)=(0,0,0); for (@gg){ $O+=$so{$_}; $P+=$sp{$_}; $R+=$sr{$_} } $O/=$N; $P/=$N; $R/=$N;
            my ($vo,$vr)=(0,0); for (@gg){ $vo+=($so{$_}-$cnt{$_}*$O)**2; $vr+=($sr{$_}-$cnt{$_}*$R)**2 } my $G=@gg; my $cf=$G>1?$G/($G-1):0;
            ($O,sqrt($vo*$cf)/$N,$P,$R,sqrt($vr*$cf)/$N,$N,$G) };
        my ($O,$seO,undef,undef,undef,$N,$G)=$stat->($LAWS{T3}[0],$ctl,0);
        printf "  band %g <= y%s: %d points in %d galaxies at T3's fit (delta = %.4f, a0 = %.4f e-10)\n",$ylo,$yhi<1e8?" <= $yhi":'',$N,$G,$DMIN{x},$a0c*1e10;
        printf "    observed mean boost %+.4f +/- %.4f dex (clustered)\n",$O,$seO;
        printf "    %-22s %12s %8s   %14s %9s %7s\n",'law (own a0)','pred boost','z','own nuisances','+/-','z';
        for my $l ('T3','T11','simple','standard','sqrt-Xsat5','sqrt-Xsat') { next unless $LAWS{$l} && $LAWS{$l}[1];
            my (undef,undef,$P)=$stat->(@{$LAWS{$l}},0); my (undef,undef,undef,$R,$seR)=$stat->(@{$LAWS{$l}},1);
            printf "    %-22s %+12.4f %8.2f   %+14.4f %9.4f %7.2f\n",$l,$P,($O-$P)/$seO,$R,$seR,$seR?$R/$seR:0 }
        if ($ylo==8) { my (undef,undef,$P)=$stat->(@{$LAWS{T3}},0);
            printf "    T3's predicted boost against none (Newtonian): %.4f dex against the error %.4f, %.2f sigma\n",$P,$seO,$P/$seO }
        if ($ylo==4) { print "    (pred boost: mean log10 nu of each law at its own a0 on T3's ratios and q, z against the observed; own nuisances: mean residual at the law's own fit)\n";
            # inclination correlation of the band residual under T3's nuisances (diagnostic)
            my (@x,@y,@gi); for my $m (@mem) { my ($g,$i)=@$m; my ($Yd,$q,$Yb)=@{$ctl->{per}{$g}}[1,2,3]; my $p=$GAL{$g}[$i]; my $gb=gbar_pt($g,$p,$Yd,$Yb);
                push @x,$INC{$g}; push @y,log10($p->[5]/($q*$q))-log10($LAWS{T3}[0]->($gb/$a0c)*$gb); push @gi,$g }
            my $pear=sub{ my ($xx,$yy)=@_; my $n=@$xx; my ($mx,$my)=(0,0); $mx+=$_ for @$xx; $my+=$_ for @$yy; $mx/=$n; $my/=$n;
                my ($sxy,$sxx,$syy)=(0,0,0); for my $k (0..$n-1){ $sxy+=($xx->[$k]-$mx)*($yy->[$k]-$my); $sxx+=($xx->[$k]-$mx)**2; $syy+=($yy->[$k]-$my)**2 }
                ($sxx>0 && $syy>0) ? ($sxy/sqrt($sxx*$syy), $sxy/$sxx) : (0,0) };
            my ($r0,$b0)=$pear->(\@x,\@y); my @ug=sort keys %{{map{$_=>1}@gi}}; my %gx=map{$_=>$INC{$_}}@ug; my $cnt=0; my $NP=2000;
            for (1..$NP) { my @perm=@ug; for (my $k=$#perm;$k>0;$k--){ my $j=int(rng()*($k+1)); @perm[$k,$j]=@perm[$j,$k] }
                my %map; @map{@ug}=@gx{@perm}; my @xp=map{$map{$_}}@gi; my ($rp)=$pear->(\@xp,\@y); $cnt++ if abs($rp)>=abs($r0) }
            my ($s1,$s2,$u1,$u2)=(0,0,0,0); my $NBB=2000; my %idx; push @{$idx{$gi[$_]}},$_ for 0..$#gi;
            for (1..$NBB) { my (@xb,@yb); for (1..@ug) { my $g=$ug[int(rng()*@ug)]; push @xb,@x[@{$idx{$g}}]; push @yb,@y[@{$idx{$g}}] }
                my ($rb,$bb)=$pear->(\@xb,\@yb); $s1+=$rb; $s2+=$rb*$rb; $u1+=$bb; $u2+=$bb*$bb }
            printf "    inclination correlation of the band residual (T3's nuisances): r = %+.3f +/- %.3f, slope %+.5f +/- %.5f dex/deg, permutation p = %.4f (%d galaxies)\n",
                $r0,sqrt($s2/$NBB-($s1/$NBB)**2),$b0,sqrt($u2/$NBB-($u1/$NBB)**2),($cnt+1)/($NP+1),scalar(@ug);
            printf "    (r: Pearson correlation of the point residuals with the catalog inclination; +/-: %d galaxy-bootstrap draws; p: %d permutations of the inclinations among the galaxies)\n",$NBB,$NP }
    }
}

# ================= J. scatter, mass-to-light spreads and points used =================
if ($SEC{J}) {
    print "\n=== J. scatter, mass-to-light spreads, points and fit quality at the best fits ===\n";
    my $nb=$FREEB?$NBUL:0;
    printf "  %-22s %9s %8s %8s %9s %9s %9s %7s %10s\n",'law','scat(dex)','sd lgYd','sd lgYb','med Yb','med Yb/Yd','chi2','pts','chi2/dof';
    my @L=(['simple','-',$fam{'simple'},1],['standard','-',$fam{standard},1],['T11','-',$fam{T11},1],['bcf','-',$fam{bcf},1]);
    push @L,['T3*',$DMIN{x},law_nuf('T3',$DMIN{x}),2] if $DMIN{x};
    for my $L (@L) { my ($l,$n,$nuf,$k)=@$L; my $bt=$BESTFIT{$l}{$n} or next; my %bY=map{$_=>$bt->{per}{$_}[1]}@GNAMES; my %bq=map{$_=>$bt->{per}{$_}[2]}@GNAMES; my %bB=map{$_=>$bt->{per}{$_}[3]}@GNAMES;
        my ($sc,undef,$np)=scatter_dex($bt->{a0},$nuf,\%bY,\%bq,\%bB);
        my @ly=map{log10($bY{$_})}@GNAMES; my $m=0; $m+=$_ for @ly; $m/=@ly; my $sd=0; $sd+=($_-$m)**2 for @ly; $sd=sqrt($sd/@ly);
        my @lb=map{log10($bB{$_})} grep{$BUL{$_}}@GNAMES; my $mb=0; $mb+=$_ for @lb; $mb/=@lb if @lb; my $sb=0; $sb+=($_-$mb)**2 for @lb; $sb=@lb?sqrt($sb/@lb):0;
        my @yb=sort{$a<=>$b} map{$bB{$_}} grep{$BUL{$_}}@GNAMES; my @rt=sort{$a<=>$b} map{$bB{$_}/$bY{$_}} grep{$BUL{$_}}@GNAMES;
        my $nu=0; $nu+=$bt->{per}{$_}[4] for @GNAMES; my $dof=$npts-3*$ngal-$nb-$k;
        printf "  %-22s %9.4f %8.3f %8.3f %9.3f %9.3f %9.2f %7d %10.4f\n",($l eq 'T3*'?"T3 $n":$l),$sc,$sd,$sb,@yb?$yb[int(@yb/2)]:0,@rt?$rt[int(@rt/2)]:0,$bt->{chi2},$nu,$bt->{chi2}/$dof }
    printf "  (dof = %d points - 3 x %d galaxies - %d free bulge ratios - global constants; scatter: rms of log10(g_obs/q^2) - log10 g_law;\n",$npts,$ngal,$nb;
    print  "   pts: points with g_bar > 0 at the fit; sd lgYd, sd lgYb: spreads of the fitted log10 ratios)\n";
}

# ================= K. robustness: the T3 scan at the wide mass-to-light prior =================
my %WIDE;
if (!$FROZEN && $SEC{K} && $OPT{wide} && $PRIOR<$OPT{wide}-1e-9 && defined $dmin) {
    print "\n=== K. robustness: the T3 scan at the wide mass-to-light prior ===\n";
    my $wp=$OPT{wide}; my $wb=$OPT{priorb} ne '' ? $OPT{priorb} : $wp;
    printf "  disk prior %.2f dex%s (bulge prior %.2f dex); same machinery\n",$wp,'',$wb;
    printf "  %-8s %8s %11s %10s %8s\n",'delta','lam','a0','chi2','D@67.4';
    my ($wd,$ws,$wa);
    { local $PRIOR=$wp; local $PRIORB=$FREEB?$wb:$wp;
      my @g=map{ sprintf('%.4f',$dmin+$_)+0 } (-0.10,-0.05,0,0.05,0.10); @g=grep{$_>0.02}@g;
      my $res=t3_scan(\@g,'T3wide',$LA_T3,1);
      my @s=sort{$a->[2]<=>$b->[2]} @$res; my @n3=sort{$a->[0]<=>$b->[0]} @s[0..2]; my ($xm,$sg,$ym)=parab3([map{[$_->[0],$_->[2]]}@n3]);
      if (defined $xm) { my $xk=sprintf('%.4f',$xm)+0; my ($a0w)=fit_a0(law_nuf('T3',$xk),log10($res->[0][1]),'T3wide*',"$xk",$xk); ($wd,$ws,$wa)=($xm,$sg,$a0w);
          printf "\n  parabolic: delta_fv = %.4f +/- %.4f (stat, Dchi2=1)\n",$xm,$sg;
          printf "  scale relation at the minimum: D\@67.4=%.3f D\@73=%.3f  product D ln(1+delta): %.3f / %.3f (a0 %.4e refitted at %.4f)\n",cH0(67.4)/$a0w,cH0(73.0)/$a0w,
              cH0(67.4)/$a0w*log(1+$xm),cH0(73.0)/$a0w*log(1+$xm),$a0w,$xk } }
    if (defined $wd) { my $sy=abs($dmin-$wd); my $tot=sqrt($sig*$sig+$sy*$sy);
        my @k=map{ $_->[0]*log(1+$_->[1]) } ([cH0(67.4)/$a0min,$dmin],[cH0(73.0)/$a0min,$dmin],[cH0(67.4)/$wa,$wd],[cH0(73.0)/$wa,$wd]);
        my @D=(cH0(67.4)/$a0min,cH0(73.0)/$a0min,cH0(67.4)/$wa,cH0(73.0)/$wa); my @ks=sort{$a<=>$b}@k; my @Ds=sort{$a<=>$b}@D;
        print "\n  combined statement:\n";
        printf "    delta_fv = %.3f +/- %.3f (stat) +/- %.3f (prior systematic: centre %.4f -> %.4f as the prior widens %.2f -> %.2f dex); total %.3f\n",$dmin,$sig,$sy,$dmin,$wd,$PRIOR,$wp,$tot;
        printf "    the product D ln(1+delta) spans %.3f to %.3f across (H0 = 67.4 to 73) x (prior %.2f to %.2f dex); measured D %.2f to %.2f\n",$ks[0],$ks[-1],$PRIOR,$wp,$Ds[0],$Ds[-1];
        printf "    derived D = 1/ln(1+delta) = %.2f +/- %.2f with the total error; A = 2/delta^2 = %.1f, band %.1f to %.1f over delta -/+ total error\n",
            1/log(1+$dmin),$tot/((1+$dmin)*log(1+$dmin)**2),2/$dmin**2,2/($dmin+$tot)**2,2/($dmin-$tot)**2;
        printf "    delta = 0.1725: %.2f sigma (stat) from the narrow-prior centre, %.2f from the wide-prior one, %.2f with the total error\n",
            abs($dmin-0.1725)/$sig,abs($wd-0.1725)/$ws,abs($dmin-0.1725)/$tot;
        %WIDE=(d=>$wd,s=>$ws,a0=>$wa) }
}

# ================= N. the knee fit (catalog distances only) =================
if ($SEC{N} && $FROZEN && defined $dmin) {
    print "\n=== N. the knee fit: MOND's simple function on the points with y > 2 at the prior centres of the ratios (y with a0 = 1.2e-10) ===\n";
    my $AREF=1.2e-10; my %keep; my $nk=0;
    for my $g (@GNAMES) { my @k; for my $p (@{$GAL{$g}}) { push @k,(($p->[0]+$Y0*$p->[1])/$AREF>2.0)?1:0 } $keep{$g}=\@k; $nk+=$_ for @k }
    my %SAVE=%GAL; my %SAVEF=%FG;
    for my $g (@GNAMES) { my @pts=map{$GAL{$g}[$_]} grep{$keep{$g}[$_]} 0..$#{$GAL{$g}}; $GAL{$g}=\@pts;
        my %a; for my $p (@pts) { push @{$a{gg}},$p->[0]; push @{$a{gs}},$p->[1]; push @{$a{Vo}},$p->[2]; push @{$a{w}},$p->[3]; push @{$a{sR}},$p->[4]; push @{$a{gd}},$p->[6]; push @{$a{gb}},$p->[7] }
        $a{$_}//=[] for qw(gg gs Vo w sR gd gb); $FG{$g}=\%a }
    my ($ak,$ck)=fit_a0($fam{'simple'},log10(1.12e-10),'knee','-');
    %GAL=%SAVE; %FG=%SAVEF;
    my $lam=log(1+$dmin);
    printf "  %d points; a0_knee = %.4e +/- %.4e (stat); D = %.3f (67.4), %.3f (73.0)\n",$nk,$ak,$BESTFIT{knee}{'-'}{sig_a0},cH0(67.4)/$ak,cH0(73.0)/$ak;
    printf "  the product with the knee scale, D(a0_knee) ln(1+delta) at T3's minimum delta = %.4f: %.3f (67.4), %.3f (73.0)\n",$dmin,cH0(67.4)/$ak*$lam,cH0(73.0)/$ak*$lam;
    printf "  the product at T3's minimum, D(a0 = %.4e) ln(1+delta): %.3f (67.4), %.3f (73.0)\n",$a0min,cH0(67.4)/$a0min*$lam,cH0(73.0)/$a0min*$lam;
    print  "  (the knee fit selects the points at the prior centres of the ratios, then fits them with the full machinery)\n";
}

# ================= L. convergence of the per-galaxy optimizer =================
if ($SEC{L}) {
    print "\n=== L. convergence of the per-galaxy optimizer ===\n";
    printf "  tolerance %.0e on the galaxy's objective per round; cap %d rounds; largest count used %d; capped fits %d of %d\n",$TOL,$CAP,$STAT{maxround},$STAT{capped},$STAT{fits};
    printf "  optimum on a bound of the box or of q in %d of %d fits; P(q) has kinks for %d galaxies and concave stretches for %d (searched piece by piece)\n",$STAT{bound},$STAT{fits},$NKINK,$NNONCONVEX;
    my @bg=sort keys %BOUNDG; printf "  galaxies with an optimum on a bound: %s\n",@bg?join(', ',map{"$_ ($BOUNDG{$_})"}@bg):'none';
    my $lawv = $BESTFIT{T11} ? ['T11','-',$fam{T11}] : undef;
    if ($lawv && $OPT{recheck}) { for my $rc (['T11',$BESTFIT{T11}{'-'},$fam{T11}],($DMIN{x}?(['T3 '.$DMIN{x},$BESTFIT{'T3*'}{$DMIN{x}},law_nuf('T3',$DMIN{x})]):())) {
        my ($nm,$bt,$nf)=@$rc; my $t0=time;
        my ($tot,undef,undef,undef,$bc)=total_chi2($bt->{a0},$nf,2); my $mx=0; my $mg='';
        for my $g (@GNAMES) { my $d=$bc->{$g}-$bt->{per}{$g}[0]; if (abs($d)>abs($mx)) { $mx=$d; $mg=$g } }
        printf "  re-check at the %s best fit with grids twice as dense (101 nodes, 41 x 41): total %.4f against %.4f (difference %+.2e); largest galaxy difference %+.2e (%s); %d s\n",
            $nm,$tot,$bt->{chi2},$tot-$bt->{chi2},$mx,$mg,time-$t0 } }
    if ($OPT{verify} && $lawv) { my $bt=$BESTFIT{T11}{'-'}; my $a0=$bt->{a0}; my $nuf=$fam{T11}; my $ia=1.0/$a0;
        print "  brute-force grid check at the T11 best fit. No optimizer is used: q is minimized at every node of the\n";
        print "  mass-to-light grid over a table of split_prior on 20,001 q nodes (401-node scan, then the\n";
        print "  201 nodes around its best); the mass-to-light grid covers the whole box (241 nodes, or 61 x 61), then\n";
        print "  seven zooms (21 or 21 x 21 nodes over +/-3 previous spacings) around each of the three lowest grid minima.\n";
        printf "    %-10s %6s %12s %26s %12s %26s %10s\n",'galaxy','bulge','converged','(Yd, Yb, q)','grid','(Yd, Yb, q)','difference';
        for my $g (split /,/,$OPT{verify}) { next unless $GAL{$g}; my $t0=time; my ($ic,$si,$sf)=@{$NUI{$g}}; my ($qlo,$qhi)=$FROZEN?(1,1):qbounds($ic,$si,$sf);
            my $two=two_ratio($g); my $NF=$FROZEN?0:20000;
            my @qf=map{$qlo+($qhi-$qlo)*$_/($NF||1)}0..$NF; my @Pf=$FROZEN?(0):map{split_prior($_,$ic,$si,$sf)}@qf;
            my $qmin=sub{ my ($op,$pp)=@_; return (0,1.0) if $FROZEN; my ($bk,$bv)=(0,1e300);
                for (my $k=0;$k<=$NF;$k+=50) { my $q=$qf[$k]; my $h=$pp*$q*$q-2*$op*$q+$Pf[$k]; ($bk,$bv)=($k,$h) if $h<$bv }
                my $ka=$bk-100; $ka=0 if $ka<0; my $kb=$bk+100; $kb=$NF if $kb>$NF;
                for my $k ($ka..$kb) { my $q=$qf[$k]; my $h=$pp*$q*$q-2*$op*$q+$Pf[$k]; ($bk,$bv)=($k,$h) if $h<$bv }
                ($bv,$qf[$bk]) };
            my $Fuv=sub{ my ($u,$v)=@_; my ($oo,$op,$pp)=suff($g,$ia,10**$u,($two?10**$v:undef),$nuf);
                my $pr=(($u-$LGY0)/$PRIOR)**2; $pr+=(($v-$LGYB0)/$PRIORB)**2 if $two; my ($h,$q)=$qmin->($op,$pp);
                ($FROZEN ? $oo-2*$op+$pp+$pr : $oo+$h+$pr, $q) };
            my ($nu,$nv)=$two?(60,60):(240,0); my %F0;
            for my $i (0..$nu) { for my $j (0..$nv) { my $u=$ULO+($UHI-$ULO)*$i/$nu; my $v=$two?$ULO+($UHI-$ULO)*$j/$nv:undef; $F0{"$i,$j"}=[$Fuv->($u,$v),$u,$v] } }
            my @lm; for my $i (0..$nu) { for my $j (0..$nv) { my $c=$F0{"$i,$j"}[0]; my $ok=1;
                for my $di (-1,0,1) { for my $dj ($two?(-1,0,1):(0)) { next if !$di && !$dj; my $nb=$F0{($i+$di).",".($j+$dj)}; $ok=0 if $nb && $nb->[0]<$c } } push @lm,$F0{"$i,$j"} if $ok } }
            @lm=(sort{$a->[0]<=>$b->[0]}@lm)[0..($#lm<2?$#lm:2)]; my $best=[1e300];
            for my $st (@lm) { my ($f,$q,$u,$v)=@$st; my ($hu,$hv)=(($UHI-$ULO)/$nu,$two?($UHI-$ULO)/$nv:0);
                for my $lev (1..7) { my $n=20; my ($bf,$bq,$bu,$bv2)=($f,$q,$u,$v);
                    for my $i (0..$n) { my $uu=$u-3*$hu+6*$hu*$i/$n; next if $uu<$ULO||$uu>$UHI;
                        for my $j (0..($two?$n:0)) { my $vv=$two?$v-3*$hv+6*$hv*$j/$n:undef; next if $two && ($vv<$ULO||$vv>$UHI);
                            my ($ff,$qq)=$Fuv->($uu,$vv); ($bf,$bq,$bu,$bv2)=($ff,$qq,$uu,$vv) if $ff<$bf } }
                    ($f,$q,$u,$v)=($bf,$bq,$bu,$bv2); $hu*=0.3; $hv*=0.3 }
                $best=[$f,$u,$v,$q] if $f<$best->[0] }
            my ($fc,$Yd,$q,$Yb)=@{$bt->{per}{$g}};
            my $fg=rec_obj($g,$a0,$nuf,10**$best->[1],$two?10**$best->[2]:$BRAT*10**$best->[1],$best->[3]);
            printf "    %-10s %6s %12.5f %26s %12.5f %26s %+10.5f   (%d s)\n",$g,$BUL{$g}?'yes':'no',$fc,sprintf('(%.4f, %.4f, %.4f)',$Yd,$Yb,$q),$fg,
                sprintf('(%.4f, %.4f, %.4f)',10**$best->[1],$two?10**$best->[2]:$BRAT*10**$best->[1],$best->[3]),$fg-$fc,time-$t0 }
        print "    (both columns are the objective with split_prior at the stated nuisances; positive difference = grid above the optimizer)\n" }
    printf "  per-galaxy objective evaluations %d; run time %d s\n",$STAT{evals},time-$T0;
}

# ================= M. the product's galaxy-clustered error =================
if ($SEC{M} && @KB) {
    print "\n=== M. the product D ln(1+delta_fv), galaxy-clustered ===\n";
    printf "  the same %d galaxy-bootstrap draws as section F: in each draw T3's minimum delta_fv comes from the\n",scalar(@KB);
    print  "  three-lowest-node parabola of the draw's profile, and log10 a0 at that delta from the draw's own per-node\n";
    print  "  a0 minima (parabola in log10 a0 at each node, interpolated through the three nodes); product = (cH0/a0) ln(1+delta), and Paper I's, with cH_dS, is 0.8275 times it\n";
    printf "  full sample by the same rule: delta_fv = %.4f, a0 = %.4f e-10, product = %.3f (67.4), %.3f (73.0)\n",$KBC{d},$KBC{a0}*1e10,$KBC{k67},$KBC{k73} if %KBC;
    my $st=sub{ my @x=sort{$a<=>$b}@_; my $n=@x; my $m=0; $m+=$_ for @x; $m/=$n; my $v=0; $v+=($_-$m)**2 for @x; $v=sqrt($v/($n-1));
        ($m,$v,$x[int(0.5*$n)],$x[int(0.16*$n)],$x[int(0.84*$n)],$x[int(0.025*$n)],$x[int(0.975*$n)],(grep{$_<=1}@x)/$n) };
    for my $c ([2,'product at H0 = 67.4'],[3,'product at H0 = 73.0'],[0,'delta_fv'],[1,'a0 at the delta minimum (e-10)']) {
        my ($k,$lab)=@$c; my @v=map{ $k==1 ? $_->[$k]*1e10 : $_->[$k] } @KB; my ($m,$sd,$md,$p16,$p84,$p025,$p975,$fl)=$st->(@v);
        printf "  %-32s mean %.4f  SD %.4f  median %.4f  16-84 percent %.4f to %.4f  2.5-97.5 %.4f to %.4f%s\n",$lab,$m,$sd,$md,$p16,$p84,$p025,$p975,
            ($k>=2?sprintf('  draws at or below 1: %d of %d (%.4f)',$fl*@v+0.5,scalar(@v),$fl):'') }
    for my $c ([2,67.4,'k67'],[3,73.0,'k73']) { my ($k,$H,$kk)=@$c; my @v=map{$_->[$k]}@KB; my ($m,$sd)=$st->(@v);
        printf "  H0 = %.1f: product %.3f, unity %.2f clustered SD below it\n",$H,$KBC{$kk},($KBC{$kk}-1)/$sd if %KBC }
}

print "\ndone.\n";
