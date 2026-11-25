#!/usr/bin/perl
# lattice_sim_pair.pl: the lattice simulation of the avalanche kinetics (avalanche/lattice/lattice_sim.pl)
# with pair-specific turning holes (Paper II, Appendix C).
#
# What it computes. In lattice_sim.pl every repack of a unit shape needs the same hole, the activation constant
# ca. Here the hole depends on the move. The file given by --holes holds the q x q matrix H[a][b], the fewest
# fresh sites a unit in orientation a needs to re-lay itself in orientation b, divided by its site count
# (written by gen_holes.py). A repack turns the site to its neighbours' plurality label, or keeps its label on
# a tie, and is activated at that move's own hole: H[a][b] for a turn, and for a repack that keeps the label
# the hole set by --same, the smallest (min), the mean (mean) or the largest (max) of the row's turning holes.
# The trigger is sampled by thinning: a kick is first tested at the cheapest hole c_min of the whole matrix,
# with the cached factor exp(-c_min/d_j), and a candidate is then kept with the probability at the move's own
# hole. --ca sets only the cold start, d_i = min(ca/8, 0.8 d0). Everything else, the options, the grids and
# the output columns, is as in lattice_sim.pl, whose header describes the model. With a matrix whose entries
# are all equal to ca (holes-rod4.txt) this script reproduces lattice_sim.pl bit for bit.
# Added output columns: holes (the file name), same, and c_min, c_mean, c_max (the cheapest hole, the mean and
# the largest of the turning holes).
#
# Paper II results it supports (App. C): only the rod of four sites lands inside the target divisor window
# 5.70 to 6.19 under every definition of the turning hole (rod4-mean: D = 5.802); four other shapes enter under
# minimum-cost reorientation (--same min), and still others under the largest change (--same max) or a plain
# mean (--same mean). summarize.py tabulates the runs.
#
# Usage: perl lattice_sim_pair.pl [the options of lattice_sim.pl] --holes FILE --same min|mean|max
#
# Commands (from the repository root; run after gen_holes.py). Each committed run file records its own inputs,
# and this loop reruns all 150 (7.4 h of recorded run time, 37 s to 9 min each), writing *.new beside each:
#   d=avalanche/unit-shapes/pair-holes
#   for f in $d/*-eta*.tsv; do
#     set -- $(awk -F'\t' 'NR == 2 {print $5, $6, $7, $8, $13, $(NF-4), $(NF-3)}' "$f")
#     perl $d/lattice_sim_pair.pl --header --L 16 --q $1 --d0 $2 --ca $3 --eta $4 --th $5 --seed 42 \
#       --holes $d/$6 --same $7 > "$f.new"; done
# Perl 5.20 or later reproduces every column but t_s; the C library's printf can round a near-tie in the
# sixth significant digit differently. The run files are <shape>-<same>-eta<eta>.tsv at th = 0.01, and
# p5-11-mean-th<th>-eta<eta>.tsv at th = 0.003 and 0.03.
use strict; use warnings;
use Getopt::Long;
use Time::HiRes qw(time);
$| = 1;

my %o = (L=>16, q=>6, d0=>0.283, ca=>1.2, eta=>0.02, dreg=>0.0, deliv=>'nn', labels=>'on', k=>6, th=>0.01,
         Z=>0.5, eps=>0.0, seed=>42, burn=>0, burncap=>0, M=>20000, rbudget=>0, nsnap=>10, smax=>2.0e6,
         header=>0, grid=>'', slice=>0, nslice=>1, list=>0, holes=>'', same=>'mean');
GetOptions(\%o, 'L=i','q=i','d0=f','ca=f','eta=f','dreg=f','deliv=s','labels=s','k=i','th=f','Z=f','eps=f',
           'seed=i','burn=i','burncap=f','M=i','rbudget=f','nsnap=i','smax=f','header!','grid=s','slice=i',
           'nslice=i','list!','idx=s','holes=s','same=s') or die "bad options\n";

my @COLS = qw(grid idx L N q d0 ca eta dreg deliv labels k th eps seed burn M t_s overflow
  A seA seA_b A_h1 A_h2 s_max sc_mom sc_mom_A2 n_p n_r nb_tot
  d_emit d_bar d_snap d_live f_dead f_loose rep_loose d_rec f_rec dw_mean f_rw
  A_B1 A_toy A_ex drift Ad2 etaAd2 D_emit
  wall wall0 G1 G2 G4 GLh xi_int mP ndom wdom fmaxdom ndom0 wdom0
  wall_f wall_l mP_f mP_l wdom_f wdom_l
  P0 P1 R0 R1 dbar0 dbar1 demit0 demit1 dold0 dold1 dw0 dw1 g_E g_R dS0 dS1
  nd_mean nd_c nd_max self_mean hist holes same c_min c_mean c_max);

sub run_one {
    my %p = (%o, @_);
    my ($L,$q,$d0,$ca,$eta,$dreg,$deliv,$labels,$k,$th,$Z,$eps,$seed) =
        @p{qw(L q d0 ca eta dreg deliv labels k th Z eps seed)};
    die "nn delivery needs k = 6\n" if $deliv eq 'nn' && $k != 6;
    my $N  = $L*$L*$L;
    my $Lh = int($L/2);
    # ---- the pair-specific holes: H[a*q+b] for a re-laying from label a to label b; Hs[a] when it keeps a ----
    my @H = ($ca) x ($q*$q);
    if ($p{holes}) {
        open(my $fh, '<', $p{holes}) or die "cannot read $p{holes}\n";
        my @v = map { split ' ' } grep { !/^\s*#/ } <$fh>; close $fh;
        die "holes file has ".scalar(@v)." numbers, q*q = ".($q*$q)."\n" unless @v == $q*$q;
        @H = @v;
    }
    my @Hs;
    for my $a (0..$q-1) {
        my @o = map { $H[$a*$q+$_] } grep { $_ != $a } 0..$q-1;
        my ($mn, $mx, $sm) = ($o[0], $o[0], 0); for (@o) { $mn = $_ if $_ < $mn; $mx = $_ if $_ > $mx; $sm += $_ }
        $Hs[$a] = $p{same} eq 'min' ? $mn : $p{same} eq 'max' ? $mx : $p{same} eq 'mean' ? $sm/@o : $p{same};
    }
    my $cmin = $Hs[0];
    for my $a (0..$q-1) { $cmin = $Hs[$a] if $Hs[$a] < $cmin; for my $b (0..$q-1) { next if $a == $b; $cmin = $H[$a*$q+$b] if $H[$a*$q+$b] < $cmin } }
    my ($csum, $cn, $cmax) = (0, 0, $cmin);
    for my $a (0..$q-1) { for my $b (0..$q-1) { next if $a == $b; $csum += $H[$a*$q+$b]; $cn++; $cmax = $H[$a*$q+$b] if $H[$a*$q+$b] > $cmax } }
    my $burn = $p{burn} > 0 ? $p{burn} : ($N*10 > 40000 ? $N*10 : 40000);
    my $M  = $p{M};
    my $SMAX = $p{smax};
    srand($seed);
    my $t0 = time;

    # ---- lattice: neighbour table (+x, -x, +y, -y, +z, -z), half index, coordinates ----
    my @nb = (0) x (6*$N);
    my @hx = (0) x $N;
    for my $z (0..$L-1) { for my $y (0..$L-1) { for my $x (0..$L-1) {
        my $i = $x + $L*($y + $L*$z); my $b = 6*$i;
        $nb[$b]   = (($x+1)%$L)    + $L*($y + $L*$z);
        $nb[$b+1] = (($x-1+$L)%$L) + $L*($y + $L*$z);
        $nb[$b+2] = $x + $L*((($y+1)%$L)    + $L*$z);
        $nb[$b+3] = $x + $L*((($y-1+$L)%$L) + $L*$z);
        $nb[$b+4] = $x + $L*($y + $L*(($z+1)%$L));
        $nb[$b+5] = $x + $L*($y + $L*(($z-1+$L)%$L));
        $hx[$i] = ($x < $L/2) ? 0 : 1;
    }}}

    # ---- cold start (subcritical, as in the toy): d_i = min(ca/8, 0.8 d0), labels random ----
    my $dinit = $ca/8.0; $dinit = 0.8*$d0 if $dinit > 0.8*$d0;
    my @d   = ($dinit) x $N;
    my @g   = (exp(-$cmin/$dinit)) x $N;      # cached bound exp(-c_min/d_i) at the cheapest hole; 0 when d_i = 0
    my @s   = map { int(rand($q)) } 1..$N;    # labels
    my @rec = (0) x $N;                       # rewritten at least once (the toy's recycled flag)
    my @cnt = (0) x $q;                       # scratch counts for the plurality label
    my @vis = (0) x $N;                       # avalanche id of a site's last repack (distinct-site count)
    my $aid = 0;

    my $theta0 = $th * $Z * $d0 * $d0 / $k;   # trigger scale, as in the toy
    my $inv_theta0 = 1.0 / $theta0;
    my $w0 = $Z * $d0 * $d0 / $k;             # primary kick
    my $shrink = 1.0 - $eta;
    my $k2 = 2*$k;                            # stack slots taken by the primary's kicks
    my $loose_thr = 0.9*$d0;                  # "fresh-loose" site (the toy's f_loose convention)
    my $lab_on = ($labels eq 'on') ? 1 : 0;
    my $nn = ($deliv eq 'nn') ? 1 : 0;
    my $inv_ln2 = 1.0/log(2.0);

    # ---- accumulators (reset after burn-in) ----
    my ($demit, $nemit, $nloose, $npc, $sum_dold, $sum_dw, $nrw) = (0,0,0,0,0,0,0);
    my (@Ph, @Rh, @demh, @doldh, @dwh);
    my $reset = sub { ($demit,$nemit,$nloose,$npc,$sum_dold,$sum_dw,$nrw) = (0,0,0,0,0,0,0);
                      @Ph=(0,0); @Rh=(0,0); @demh=(0,0); @doldh=(0,0); @dwh=(0,0); };
    $reset->();

    # ---- one avalanche: primary, then drain the cascade (LIFO flat stack of (site, kick) pairs) ----
    my $avalanche = sub {
        my $i;
        if ($eps) {
            my $h = (rand() < 0.5*(1.0+$eps)) ? 0 : 1;
            my $x = $h*$Lh + int(rand($Lh));
            $i = $x + $L*(int(rand($L)) + $L*int(rand($L)));
        } else {
            $i = int(rand($N));
        }
        my $hi = $hx[$i];
        my $dold = $d[$i];
        my $lab = int(rand($q));                       # the roughening event lays a random label
        $s[$i] = $lab;
        my $b = 6*$i;
        my ($best, $bc, $tie) = (-1, 0, 0);
        for my $l (@s[@nb[$b..$b+5]]) {
            my $c = ++$cnt[$l];
            if ($c > $bc) { $bc = $c; $best = $l; $tie = 0 } elsif ($c == $bc) { $tie = 1 }
        }
        $cnt[$_] = 0 for @s[@nb[$b..$b+5]];
        my $inreg = (!$tie && $best == $lab) ? 1 : 0;   # aligned with its neighbours?
        my $dw = ($lab_on && $inreg) ? $dreg : $d0;      # wall site: laid loose; aligned: dreg
        $d[$i] = $dw; $g[$i] = $dw > 0 ? exp(-$cmin/$dw) : 0; $rec[$i] = 1;
        $sum_dold += $dold; $sum_dw += $dw; $nrw += $inreg;
        $Ph[$hi]++; $doldh[$hi] += $dold; $dwh[$hi] += $dw;
        my @q;
        if ($nn) { push @q, $nb[$b],$w0, $nb[$b+1],$w0, $nb[$b+2],$w0, $nb[$b+3],$w0, $nb[$b+4],$w0, $nb[$b+5],$w0; }
        else     { push @q, int(rand($N)), $w0 for 1..$k; }
        my $sz = 1;
        my ($nd, $nself) = (0, 0);                                  # distinct sites repacked; repacks of site i
        $aid++;
        while (@q) {
            my $wk = pop @q; my $j = pop @q;
            my $u = rand();
            next if $u >= $wk * $g[$j] * $inv_theta0;              # bound at the cheapest hole
            my $bj = 6*$j;
            ($best, $bc, $tie) = (-1, 0, 0);                        # the target: the neighbours' plurality
            for my $l (@s[@nb[$bj..$bj+5]]) {
                my $c = ++$cnt[$l];
                if ($c > $bc) { $bc = $c; $best = $l; $tie = 0 } elsif ($c == $bc) { $tie = 1 }
            }
            $cnt[$_] = 0 for @s[@nb[$bj..$bj+5]];
            my $sj = $s[$j]; my $tg = $tie ? $sj : $best;
            my $ch = ($tg == $sj) ? $Hs[$sj] : $H[$sj*$q + $tg];
            next if $ch > $cmin && $u >= $wk * exp(-$ch/$d[$j]) * $inv_theta0;   # the move's own hole
            $npc++ if @q < $k2;                                     # a kick of the primary itself
            if ($vis[$j] != $aid) { $vis[$j] = $aid; $nd++ }
            $nself++ if $j == $i;
            my $dj = $d[$j];                                        # repack: release stored misfit ...
            my $wj = $Z * $dj * $dj / $k;
            if ($nn) { push @q, $nb[$bj],$wj, $nb[$bj+1],$wj, $nb[$bj+2],$wj, $nb[$bj+3],$wj, $nb[$bj+4],$wj, $nb[$bj+5],$wj; }
            else     { push @q, int(rand($N)), $wj for 1..$k; }
            $s[$j] = $tg;                                           # ... relabel to the neighbours' plurality ...
            $demit += $dj; $nemit++; $nloose++ if $dj > $loose_thr;
            my $hj = $hx[$j]; $Rh[$hj]++; $demh[$hj] += $dj;
            my $dn = $dj * $shrink;                                 # ... then compact by the quantum eta
            $d[$j] = $dn; $g[$j] = exp(-$cmin/$dn);
            $sz++;
            return ($sz, 1, $nd, $nself) if $sz > $SMAX;
        }
        return ($sz, 0, $nd, $nself);
    };

    # ---- snapshot of the fields: free-volume statistics, wall fraction, correlation, domains ----
    my $snap = sub {
        my %r;
        my ($sd, $nlive, $sdl, $nls, $ndead) = (0,0,0,0,0);
        for my $i (0..$N-1) { my $x = $d[$i]; $sd += $x; if ($x > 0) { $nlive++; $sdl += $x; $nls++ if $x > $loose_thr } else { $ndead++ } }
        $r{d_snap} = $sd/$N; $r{d_live} = $nlive ? $sdl/$nlive : 0; $r{f_dead} = $ndead/$N; $r{f_loose} = $nls/$N;
        $r{sumd} = $sd;
        my $nwall = 0;
        for my $i (0..$N-1) {
            my $b = 6*$i; my ($best, $bc, $tie) = (-1, 0, 0);
            for my $l (@s[@nb[$b..$b+5]]) { my $c = ++$cnt[$l]; if ($c > $bc) { $bc = $c; $best = $l; $tie = 0 } elsif ($c == $bc) { $tie = 1 } }
            $cnt[$_] = 0 for @s[@nb[$b..$b+5]];
            $nwall++ unless (!$tie && $best == $s[$i]);
        }
        $r{wall} = $nwall/$N;
        my @lc = (0) x $q; $lc[$_]++ for @s; my $mx = 0; for (@lc) { $mx = $_ if $_ > $mx }
        $r{mP} = $q > 1 ? ($q*$mx/$N - 1)/($q - 1) : 1;
        my @same = (0) x ($Lh+1);
        for my $z (0..$L-1) { for my $y (0..$L-1) { my $row = $L*($y + $L*$z);
            for my $x (0..$L-1) { my $sl = $s[$row+$x];
                for my $rr (1..$Lh) {
                    $same[$rr]++ if $s[$row + (($x+$rr) % $L)] == $sl;
                    $same[$rr]++ if $s[$x + $L*((($y+$rr) % $L) + $L*$z)] == $sl;
                    $same[$rr]++ if $s[$x + $L*($y + $L*(($z+$rr) % $L))] == $sl;
                } } } }
        my @G = (0); for my $rr (1..$Lh) { my $ps = $same[$rr]/(3*$N); push @G, ($q*$ps - 1)/($q - 1) }
        $r{G} = \@G;
        my @seen = (0) x $N; my ($ncl, $s2, $smx) = (0,0,0);
        for my $i0 (0..$N-1) {
            next if $seen[$i0];
            $ncl++; my $l = $s[$i0]; my @st = ($i0); $seen[$i0] = 1; my $size = 0;
            while (@st) { my $u = pop @st; $size++; my $b = 6*$u;
                for my $v (@nb[$b..$b+5]) { if (!$seen[$v] && $s[$v] == $l) { $seen[$v] = 1; push @st, $v } } }
            $s2 += $size*$size; $smx = $size if $size > $smx;
        }
        $r{ndom} = $N/$ncl; $r{wdom} = $s2/$N; $r{fmaxdom} = $smx/$N;
        return \%r;
    };

    my $s0 = $snap->();                       # the cold start: a random label field (reference values)

    # ---- burn-in from the subcritical cold start ----
    my ($bp, $btot, $over) = (0, 0, 0);
    while ($bp < $burn && (!$p{burncap} || $btot < $p{burncap})) {
        my ($sz, $ov) = $avalanche->(); $bp++; $btot += $sz; $over += $ov;
    }
    $reset->();
    my $sumd_start = 0; $sumd_start += $_ for @d;
    my @sdh_start = (0,0); $sdh_start[$hx[$_]] += $d[$_] for 0..$N-1;

    # ---- measurement ----
    my $nsnap = $p{nsnap};
    my $every = $nsnap > 0 ? int($M/$nsnap) : 0; $every = 1 if $nsnap > 0 && $every < 1;
    my ($m, $Ssum, $S2sum, $Rtot, $smax) = (0,0,0,0,0);
    my @bsum = (0) x 10; my @bcnt = (0) x 10;
    my ($Sh1, $Mh1) = (0,0);
    my @hist;
    my %acc; my @Gacc; my $ns = 0;
    my @sdh_snap = (0,0);
    my ($NDs, $ND2s, $NDmax, $NSs) = (0, 0, 0, 0);
    while ($m < $M && (!$p{rbudget} || $Rtot < $p{rbudget})) {
        my ($sz, $ov, $nd, $nself) = $avalanche->();
        $NDs += $nd; $ND2s += $nd*$nd; $NDmax = $nd if $nd > $NDmax; $NSs += $nself;
        $over += $ov; $m++;
        $Ssum += $sz; $S2sum += $sz*$sz; $Rtot += $sz; $smax = $sz if $sz > $smax;
        my $bi = int(($m-1)*10/$M); $bi = 9 if $bi > 9; $bsum[$bi] += $sz; $bcnt[$bi]++;
        if ($m <= $M/2) { $Sh1 += $sz; $Mh1++ }
        $hist[int(4*log($sz)*$inv_ln2 + 1e-9)]++;
        if ($every && $m % $every == 0 && $ns < $nsnap) {
            my $r = $snap->(); $ns++;
            $acc{$_} += $r->{$_} for qw(d_snap d_live f_dead f_loose wall mP ndom wdom fmaxdom);
            if ($ns == 1) { $acc{"${_}_f"} = $r->{$_} for qw(wall mP wdom) }      # first and last snapshots:
            $acc{"${_}_l"} = $r->{$_} for qw(wall mP wdom);                       # stationarity of the labels
            for my $rr (1..$Lh) { $Gacc[$rr] += $r->{G}[$rr] }
            for my $i (0..$N-1) { $sdh_snap[$hx[$i]] += $d[$i] }
        }
    }
    my $t1 = time;
    my $sumd_end = 0; $sumd_end += $_ for @d;
    my @sdh_end = (0,0); $sdh_end[$hx[$_]] += $d[$_] for 0..$N-1;

    # ---- results ----
    my %r;
    @r{qw(grid idx L N q d0 ca eta dreg deliv labels k th eps seed burn M)} =
        ($p{gridname} // '-', $p{grididx} // -1, $L, $N, $q, $d0, $ca, $eta, $dreg, $deliv, $labels, $k, $th, $eps, $seed, $bp, $m);
    $r{t_s} = sprintf("%.1f", $t1 - $t0);
    $r{overflow} = $over;
    my $A = $Ssum/$m;
    my $var = $S2sum/$m - $A*$A; $var = 0 if $var < 0;
    $r{A} = $A; $r{seA} = sqrt($var/$m);
    my @bm = map { $bcnt[$_] ? $bsum[$_]/$bcnt[$_] : () } 0..9;
    my $bmean = 0; $bmean += $_ for @bm; $bmean /= @bm;
    my $bv = 0; $bv += ($_ - $bmean)**2 for @bm; $bv /= (@bm - 1) if @bm > 1;
    $r{seA_b} = sqrt($bv/@bm);
    $r{A_h1} = $Mh1 ? $Sh1/$Mh1 : 0; $r{A_h2} = ($m - $Mh1) ? ($Ssum - $Sh1)/($m - $Mh1) : 0;
    $r{s_max} = $smax;
    $r{sc_mom} = 2*$S2sum/$Ssum; $r{sc_mom_A2} = $r{sc_mom}/($A*$A);
    my $nrep = $nemit;
    $r{n_p} = $npc/$m;                                           # repacks triggered by the primary's kicks
    $r{n_r} = $nrep ? ($nrep - $npc)/$nrep : 0;                  # repacks triggered per repack
    $r{nb_tot} = ($A - 1)/$A;                                    # all repacks per event (= 1 - 1/A identically)
    my $dem = $nemit ? $demit/$nemit : 0;
    $r{d_emit} = $dem;
    my $dbar = $sum_dold/$m;                                     # population mean, sampled at the primaries
    $r{d_bar} = $dbar;
    for (qw(d_snap d_live f_dead f_loose wall mP ndom wdom fmaxdom)) { $r{$_} = $ns ? $acc{$_}/$ns : 0 }
    for (qw(wall_f wall_l mP_f mP_l wdom_f wdom_l)) { $r{$_} = $acc{$_} // 'NA' }
    $r{rep_loose} = $nemit ? $nloose/$nemit : 0;
    my ($sr, $nr) = (0, 0); for my $i (0..$N-1) { next unless $rec[$i]; $nr++; $sr += $d[$i] }
    $r{d_rec} = $nr ? $sr/$nr : 0; $r{f_rec} = $nr/$N;
    $r{dw_mean} = $sum_dw/$m; $r{f_rw} = $nrw/$m;
    $r{A_B1}  = $dbar > 0 ? 1 + ($d0 - $dbar)/($eta*$dbar) : 0;                       # cruder form
    $r{A_toy} = $dem > 0 ? 1 + ($d0 - $r{d_rec})/($eta*$dem) : 0;                     # the toy's A_flux
    $r{A_ex}  = $dem > 0 ? 1 + ($r{dw_mean} - $dbar)/($eta*$dem) : 0;                 # Eq. (15), before the drift
    $r{drift} = $dem > 0 ? ($sumd_end - $sumd_start)/($m*$eta*$dem) : 0;              # A = A_ex - drift exactly
    $r{Ad2} = $A*$dem*$dem; $r{etaAd2} = $eta*$r{Ad2};
    $r{D_emit} = $dem > 0 ? 1/log(1+$dem) : 0;
    $r{wall0} = $s0->{wall}; $r{ndom0} = $s0->{ndom}; $r{wdom0} = $s0->{wdom};
    my @G = map { $ns ? ($Gacc[$_] // 0)/$ns : 0 } 0..$Lh;
    $r{G1} = $G[1]; $r{G2} = $G[2] // 0; $r{G4} = $G[4] // 0; $r{GLh} = $G[$Lh];
    my $xi = 0; $xi += $G[$_] for 1..$Lh; $r{xi_int} = $xi;
    @r{qw(P0 P1 R0 R1)} = (@Ph, @Rh);
    $r{dbar0} = $ns ? $sdh_snap[0]/($ns*$N/2) : 0; $r{dbar1} = $ns ? $sdh_snap[1]/($ns*$N/2) : 0;
    $r{demit0} = $Rh[0] ? $demh[0]/$Rh[0] : 0; $r{demit1} = $Rh[1] ? $demh[1]/$Rh[1] : 0;
    $r{dold0} = $Ph[0] ? $doldh[0]/$Ph[0] : 0; $r{dold1} = $Ph[1] ? $doldh[1]/$Ph[1] : 0;
    $r{dw0} = $Ph[0] ? $dwh[0]/$Ph[0] : 0; $r{dw1} = $Ph[1] ? $dwh[1]/$Ph[1] : 0;
    if ($eps && $Ph[0] && $Ph[1] && $Rh[0] && $Rh[1]) {
        my $lp = log($Ph[0]/$Ph[1]);
        $r{g_E} = log(($Ph[0]+$Rh[0])/($Ph[1]+$Rh[1]))/$lp;
        $r{g_R} = log($Rh[0]/$Rh[1])/$lp;
    } else { $r{g_E} = 'NA'; $r{g_R} = 'NA' }
    $r{dS0} = $sdh_end[0] - $sdh_start[0]; $r{dS1} = $sdh_end[1] - $sdh_start[1];   # free-volume drift per half
    $r{nd_mean} = $NDs/$m;                                       # distinct sites repacked per avalanche
    $r{nd_c} = $NDs ? 2*$ND2s/$NDs : 0;                          # its moment scale 2<n_d^2>/<n_d>
    $r{nd_max} = $NDmax;
    $r{self_mean} = $NSs/$m;                                     # repacks of the primary's own site
    $r{hist} = join(",", map { defined $hist[$_] ? "$_:$hist[$_]" : () } 0..$#hist);
    ($r{holes} = $p{holes} || '-') =~ s{.*[/\\]}{}; $r{same} = $p{same};
    $r{c_min} = $cmin; $r{c_mean} = $cn ? $csum/$cn : $cmin; $r{c_max} = $cmax;
    return join("\t", map { my $v = $r{$_}; (defined $v && $v =~ /^-?\d*\.\d+(e[-+]?\d+)?$/i) ? sprintf("%.6g", $v) : (defined $v ? $v : 'NA') } @COLS);
}

# ---------------------------------------------------------------------------------------------------------
# Named grids. Each point is a hash of options overriding the defaults above. Seeds are fixed per grid.
# ---------------------------------------------------------------------------------------------------------
my %GRIDS;
{
    # validation: the toy's kernel (random delivery, labels off, k = 12) with its burn-in cap and repack budget
    my @v;
    for my $seed (42, 7, 11) {
        push @v, {L=>27, q=>6, d0=>0.30, ca=>1.2, eta=>0.015, deliv=>'rn', labels=>'off', k=>12,
                  burn=>40000, burncap=>1.5e6, M=>6000, rbudget=>1.0e6, seed=>$seed};
        push @v, {L=>37, q=>6, d0=>0.50, ca=>1.2, eta=>0.05, deliv=>'rn', labels=>'off', k=>12,
                  burn=>40000, burncap=>1.5e6, M=>16000, rbudget=>2.5e6, seed=>$seed};
    }
    $GRIDS{valid} = \@v;

    # main scan: the working size L = 16, lattice delivery, labels on, every (d0, ca, q) on one eta grid
    my @ETA = (0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05);
    my @mn;
    for my $q (2, 6, 12) { for my $d0 (0.163, 0.283, 0.334) { for my $ca (0.7, 1.2, 1.7) { for my $eta (@ETA) {
        push @mn, {L=>16, q=>$q, d0=>$d0, ca=>$ca, eta=>$eta, deliv=>'nn', labels=>'on', seed=>42};
    }}}}
    $GRIDS{main} = \@mn;

    # extension: efficiencies below the main grid, for the families whose crossing lies below eta = 0.002
    my @ex;
    for my $eta (0.0007, 0.001, 0.0014) { push @ex, {L=>16, q=>2, d0=>0.163, ca=>0.7, eta=>$eta, seed=>42} }
    for my $eta (0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0014) { push @ex, {L=>16, q=>2, d0=>0.283, ca=>1.7, eta=>$eta, seed=>42} }
    for my $q (6, 12) { for my $eta (0.001, 0.0014) { push @ex, {L=>16, q=>$q, d0=>0.283, ca=>1.7, eta=>$eta, seed=>42} } }
    $GRIDS{ext} = \@ex;

    # the two reference families: R1 = the poured packing d0 = 0.283 at ca = 1.2, R2 = the random loose
    # packing d0 = 0.334 at the top of the activation range, ca = 1.7; eta brackets their crossings
    my @REF = ({q=>6, d0=>0.283, ca=>1.2}, {q=>6, d0=>0.334, ca=>1.7});
    my @REFETA = (0.005, 0.0075, 0.01, 0.015);

    # modes: lattice against random-neighbour delivery, labels on against off (the main grid holds nn/on),
    # and the toy's kernel (random delivery, labels off, k = 12) on the same sites
    my @md;
    for my $r (@REF) { for my $eta (@REFETA) {
        push @md, {%$r, L=>16, eta=>$eta, deliv=>'nn', labels=>'off', seed=>42};
        push @md, {%$r, L=>16, eta=>$eta, deliv=>'rn', labels=>'on',  seed=>42};
        push @md, {%$r, L=>16, eta=>$eta, deliv=>'rn', labels=>'off', seed=>42};
        push @md, {%$r, L=>16, eta=>$eta, deliv=>'rn', labels=>'off', k=>12, seed=>42};
    }}
    $GRIDS{modes} = \@md;

    # finite size: L = 12, 24, 32 at the reference families (seed 42), and seed replicates at L = 16 and 32
    my @fs;
    for my $r (@REF) { for my $eta (@REFETA) {
        for my $L (12, 24, 32) { push @fs, {%$r, L=>$L, eta=>$eta, seed=>42} }
        for my $sd (7, 11)     { push @fs, {%$r, L=>16, eta=>$eta, seed=>$sd} }
    }}
    for my $r (@REF) { for my $eta (0.0075, 0.01) { push @fs, {%$r, L=>32, eta=>$eta, seed=>7} } }
    $GRIDS{fsize} = \@fs;

    # gain: drive asymmetry eps between the halves, at the eta nearest each reference crossing
    my @gn;
    for my $r (@REF) {
        for my $sd (42, 7, 11) {
            push @gn, {%$r, L=>16, eta=>0.0075, eps=>0.2, M=>60000, deliv=>'nn', seed=>$sd};
            push @gn, {%$r, L=>16, eta=>0.0075, eps=>0.2, M=>60000, deliv=>'rn', seed=>$sd};
        }
        push @gn, {%$r, L=>16, eta=>0.0075, eps=>0.4, M=>60000, deliv=>'nn', seed=>42};
        push @gn, {%$r, L=>32, eta=>0.0075, eps=>0.2, M=>60000, deliv=>'nn', seed=>42};
    }
    $GRIDS{gain} = \@gn;

    # the corner d0 = 0.5, ca = 1.2 of the toy's scan mean-field-scan-signed-modulus.tsv, on the lattice and off it, q = 6
    my @cn;
    for my $eta (0.02, 0.03, 0.05, 0.075) {
        for my $m (['nn','on',6], ['nn','off',6], ['rn','on',6], ['rn','off',6], ['rn','off',12]) {
            push @cn, {L=>16, q=>6, d0=>0.5, ca=>1.2, eta=>$eta, deliv=>$m->[0], labels=>$m->[1], k=>$m->[2], seed=>42};
        }
    }
    # the toy's kernel at the toy's own size (5e4 sites) with this script's uncapped burn-in of 10 sweeps
    push @cn, {L=>37, q=>6, d0=>0.5, ca=>1.2, eta=>0.05, deliv=>'rn', labels=>'off', k=>12, seed=>42};
    $GRIDS{corner} = \@cn;

    # sensitivity at R1: trigger scale th x(1/3) and x3, and aligned primaries laid half loose (dreg = d0/2)
    my @sn;
    for my $eta (@REFETA) {
        push @sn, {%{$REF[0]}, L=>16, eta=>$eta, th=>0.003, seed=>42};
        push @sn, {%{$REF[0]}, L=>16, eta=>$eta, th=>0.03,  seed=>42};
        push @sn, {%{$REF[0]}, L=>16, eta=>$eta, dreg=>0.1415, seed=>42};
    }
    push @sn, {%{$REF[0]}, L=>16, eta=>0.003, th=>0.03, seed=>42};    # index 12: brackets the th = 0.03 crossing
    $GRIDS{sens} = \@sn;

    # the top of the canonical Cohen-Turnbull range, ca = 1.0, for the random loose packing d0 = 0.334
    my @c1;
    for my $q (2, 6) { for my $eta (0.0075, 0.01, 0.015, 0.02, 0.03) {
        push @c1, {L=>16, q=>$q, d0=>0.334, ca=>1.0, eta=>$eta, seed=>42};
    }}
    $GRIDS{ca1} = \@c1;
}

if ($o{grid}) {
    my $g = $GRIDS{$o{grid}} or die "no grid $o{grid}; have: @{[sort keys %GRIDS]}\n";
    if ($o{list}) {
        for my $i (0..$#$g) { print "$i\t", join(" ", map { "$_=$g->[$i]{$_}" } sort keys %{$g->[$i]}), "\n" }
        exit;
    }
    # load balance: longest-processing-time-first assignment of points to slices, with the cost estimated as
    # (primaries) x (1/eta) x (snapshot weight); deterministic, so a slice always holds the same points
    my @cost = map { my %p = (%o, %{$g->[$_]}); my $N = $p{L}**3;
                     my $b = $p{burn} > 0 ? $p{burn} : ($N*10 > 40000 ? $N*10 : 40000);
                     ($b + $p{M}) / $p{eta} } 0..$#$g;
    my @order = sort { $cost[$b] <=> $cost[$a] || $a <=> $b } 0..$#$g;
    my @load = (0) x $o{nslice}; my @mine;
    for my $i (@order) {
        my $sl = 0; for my $t (1..$o{nslice}-1) { $sl = $t if $load[$t] < $load[$sl] }
        $load[$sl] += $cost[$i]; push @mine, $i if $sl == $o{slice};
    }
    @mine = split /,/, $o{idx} if defined $o{idx} && $o{idx} ne "";   # --idx i,j,k: exactly these points (same seeds)
    print join("\t", @COLS), "\n";
    my $t0 = time;
    for my $i (@mine) {
        print run_one(%{$g->[$i]}, gridname=>$o{grid}, grididx=>$i), "\n";
        warn sprintf("[%s slice %d, %.0f s] done point %d of %d\n", $o{grid}, $o{slice}, time-$t0, $i, scalar @$g);
    }
    exit;
}
print join("\t", @COLS), "\n" if $o{header};
print run_one(), "\n";
