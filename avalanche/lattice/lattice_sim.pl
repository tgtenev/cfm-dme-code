#!/usr/bin/perl
# lattice_sim.pl: the lattice simulation of the avalanche kinetics (Paper II, Sec. 5.1 and Appendix C).
#
# What it computes. The kernel of the mean-field toy (avalanche/mean_field_toy.pl, Eq. (14) of Paper II),
# placed on the L^3 sites of a simple cubic lattice with periodic boundaries and six nearest neighbours:
#   * a kick of energy w at site j triggers a repack with probability min{1, (w/theta0) exp(-ca/d_j)};
#   * a repack releases the stored misfit d_j^2, re-emits k kicks of Z d_j^2/k (a LIFO stack, as in the toy)
#     and compacts d_j -> (1 - eta) d_j;
#   * theta0 = th Z d0^2/k with th = 0.01 (Paper II: theta0 = 0.01 Z_1 d0^2/6); Z = 0.5 cancels; cold start
#     d_i = min(ca/8, 0.8 d0); a cap of 2e6 events per avalanche, with the overflows counted; s counts the
#     primary plus its repacks, and A = <s>.
# The lattice adds the orientation labels of the units:
#   * each site carries a label s_i in {0..q-1} besides its free volume d_i >= 0;
#   * a site is ALIGNED when its label equals the unique plurality label of its six neighbours; a tie for the
#     plurality (two or more labels sharing the top count) counts as not aligned; a site that is not aligned
#     is a WALL SITE, a site of a domain wall;
#   * PRIMARY (a roughening event, which lays its unit at random): a random site gets a random label; if it is
#     then a wall site it is laid loose, d_i = d0, and if it is aligned it is laid d_i = dreg (default 0)
#     (Paper II: matching neighbour plurality sets d_w = 0, otherwise d0); either way it delivers k = 6
#     kicks of w0 = Z d0^2/k to its six neighbours;
#   * a REPACK also relabels the site to the plurality label of its six neighbours (on a tie it keeps its
#     label);
#   * delivery 'nn' sends a site's k = 6 kicks to its six neighbours (Paper II, Table 1: six equal kicks to the
#     nearest neighbours, adopted in place of the long-range Eshelby kicks); delivery 'rn' sends k kicks to k
#     random sites (the toy's delivery, the random-neighbour control), the labels still living on the lattice;
#   * labels 'on' feeds the alignment back into the laying (d0 or dreg); labels 'off' lays d0 always, as the
#     toy does, while the label dynamics still run (so the label statistics remain defined).
# Other differences from the toy: the factor exp(-ca/d_j) is cached per site and updated when d_j changes
# (the same probabilities, fewer exponentials); a site with d_j = 0 never triggers; the burn-in is
# max(40000, 10 N) primaries with no cap on the repacks (the toy stops its burn-in at 1.5e6 repacks;
# --burncap restores that); the measurement runs a fixed number M of primaries (--rbudget restores the
# toy's repack budget).
#
# Paper II results it supports: the 398 lattice runs of App. C (the nine grids below: the sphere-packing
# inputs d0 = 0.163, 0.283 and 0.334, the 16^3 working size checked from 12^3 to 32^3, 2 to 12 labels and the
# random-neighbour controls), tabulated by lattice_analyze.py. Run with the inputs of the rod of four sites
# (--q 3 --d0 0.2818 --ca 0.75), it gives the free volume 0.188, A = 57, eta_a = 1.0 percent and divisor
# 5.81 of Sec. 5.1 and 5.2 and the checks of the trigger scale, the half hole and the laying rule; run with
# the inputs of every other unit shape, it gives the catalogue's divisors (see "The runs of the unit shapes").
#
# Usage:
#   perl lattice_sim.pl [options]                         one point, one TSV line (with --header, the header too)
#   perl lattice_sim.pl --grid NAME --slice I --nslice NS slice I of NS of a named grid (TSV); points are assigned
#                                                   to slices longest-first on the cost estimate (burn+M)/eta
#   perl lattice_sim.pl --grid NAME --idx i,j,k           exactly the listed points of a named grid (same seeds)
#   perl lattice_sim.pl --grid NAME --list                list a grid's points with their indices
# Grids: valid, main, ext, modes, fsize, gain, corner, sens, ca1 (definitions below; seeds fixed per point).
# Options (defaults): --L 16 --q 6 --d0 0.283 --ca 1.2 --eta 0.02 --dreg 0 --deliv nn|rn --labels on|off
#   --k 6 --th 0.01 --Z 0.5 --eps 0 (drive asymmetry between the halves x < L/2 and x >= L/2: a primary lands
#   in the first half with probability (1+eps)/2) --seed 42 --burn 0 (auto) --burncap 0 (none) --M 20000
#   --rbudget 0 (none) --nsnap 10 (snapshots of the label and free-volume fields during the measurement)
#
# Output columns. Inputs: grid and idx (the named grid and the point's index, or - and -1), L, N = L^3, q, d0,
# ca, eta, dreg, deliv, labels, k, th, eps, seed; burn and M (primaries in the burn-in and in the measurement);
# t_s (run time, s); overflow (avalanches that reached the cap).
#   A, seA (naive standard error), seA_b (batch-means standard error, 10 batches), A_h1 and A_h2 (A over the
#   two halves of the measurement), s_max; sc_mom = 2<s^2>/<s>, the moment estimate of the size cutoff, and
#   sc_mom_A2 = sc_mom/A^2; n_p (repacks triggered by the primary's own kicks, per avalanche), n_r (repacks
#   triggered per repack, the cascade's branching ratio), nb_tot = 1 - 1/A.
#   d_emit (mean free volume of the repacked sites, before the compaction); d_bar (mean free volume of the
#   sites the primaries land on, the standing mean); d_snap, d_live, f_dead, f_loose (snapshot means: the
#   free volume, the free volume of sites with d > 0, the fraction with d = 0, the fraction with d > 0.9 d0);
#   rep_loose (fraction of repacks at sites with d > 0.9 d0); d_rec and f_rec (as in the toy); dw_mean (the
#   mean laid free volume <d_w>); f_rw (fraction of primaries laid aligned).
#   Forms of the flux balance: A_ex = 1 + (<d_w> - d_bar)/(eta d_emit), Eq. (15) of Paper II; drift, the
#   change of the total free volume over the measurement over (M eta d_emit), so that A = A_ex - drift
#   exactly; A_toy = 1 + (d0 - d_rec)/(eta d_emit), the toy's A_flux; A_B1 = 1 + (d0 - d_bar)/(eta d_bar), a
#   cruder form with d0 in place of <d_w> and d_bar in place of d_emit. Ad2 = A d_emit^2, etaAd2 = eta Ad2,
#   D_emit = 1/ln(1 + d_emit), the divisor at Z_1 = 1.
#   Labels: wall (wall-site fraction, snapshot mean), wall0 (at the cold start), wall_f and wall_l (first and
#   last snapshots); G1, G2, G4, GLh (label correlation at distance 1, 2, 4 and L/2) and xi_int (its sum over
#   distance); mP (order parameter of the most common label); ndom, wdom, fmaxdom (mean and weight-averaged
#   domain size, largest domain's fraction), ndom0 and wdom0 at the cold start; mP_f, mP_l, wdom_f, wdom_l.
#   Halves x < L/2 (0) and x >= L/2 (1): P0, P1 (primaries) and R0, R1 (repacks); dbar0, dbar1, demit0,
#   demit1, dold0, dold1, dw0, dw1 (per-half means); g_E and g_R (gain of all events and of repacks between
#   the halves, NA when eps = 0); dS0, dS1 (free-volume change per half).
#   nd_mean, nd_c, nd_max (distinct sites repacked per avalanche: mean, 2<n^2>/<n>, largest); self_mean
#   (repacks of the primary's own site); hist (avalanche sizes, four bins per octave, bin = int(4 log2 s)).
#
# Commands (from the repository root). The 398 runs of the nine grids, 6.7 h of single-process run time in
# all (the longest run 9 min), parallel with --slice I --nslice N:
#   for g in valid main ext modes fsize gain corner sens ca1; do
#     perl avalanche/lattice/lattice_sim.pl --grid $g > avalanche/lattice/run-$g-0.tsv; done
# The committed files run-<grid>-<i>.tsv hold the same points split over parallel slices and --idx lists;
# remove them before a rerun. Each point seeds its own generator, so a row does not depend on the split.
# Perl 5.20 or later reproduces every column but t_s; the C library's printf can round a near-tie in the
# sixth significant digit differently. Seven committed files, run-main-0.tsv to run-main-5.tsv and
# run-valid-0.tsv (183 rows), lack the six columns dS0, dS1, nd_mean, nd_c, nd_max and self_mean; a rerun adds
# them, and lattice_analyze.py then prints values where its tables now show n/a for those rows.
# Run order: first, then lattice_analyze.py, then rebuild_crossings.py.
#
# The runs of the unit shapes (from the repository root; L = 16, seed 42, one TSV row each; run after
# latticebricks.py and polycubes.py, which give d0 and ca, and before catalogue/build_results.py):
#   the rod series, 42 runs, with (D0, CA) = (0.0893, 0.5), (0.1922, 0.6667), (0.2818, 0.75), (0.3558, 0.8),
#   (0.4275, 0.8333), (0.5408, 0.875) for L = 2, 3, 4, 5, 6, 8 (the means of lattice-L<L>-s*.tsv), and
#   ETA = 0.004, 0.006, 0.008, 0.011, 0.015, 0.02, 0.027:
#     perl avalanche/lattice/lattice_sim.pl --header --L 16 --q 3 --d0 D0 --ca CA --eta ETA --seed 42 \
#       > avalanche/unit-shapes/rod/L<L>-eta<ETA>.tsv
#   the checks, 42 runs into avalanche/unit-shapes/checks/, the same command with
#     --q 3 --d0 0.2818 --ca 0.75 --th 0.003 or 0.03 at the rod's seven ETA       (L4-th<th>-eta<ETA>.tsv)
#     --q 3 --d0 0.2818 --ca 0.375 --th 0.01, ETA = 0.008 0.011 0.015 0.02 0.027 0.037 0.05, half the hole
#                                                                                 (L4-th0.01-ca0.375-eta<ETA>.tsv)
#     --q 3 --d0 0.4909 --ca 0.75 --th 0.01, ETA = 0.015 0.02 0.027 0.037 0.05 0.07 0.1, the one-draw front
#                                                                                 (L4front1-th0.01-eta<ETA>.tsv)
#     --q 6 --d0 0.3547 --ca 0.5 --th 0.003 or 0.03, ETA = 0.0053 0.008 0.0115 0.0177 0.0266 0.039 0.0585,
#     the 1 x 2 x 4 brick                                                         (brick124-th<th>-eta<ETA>.tsv)
#   the catalogue, 392 runs avalanche/unit-shapes/catalogue/<shape>-eta<ETA>.tsv, with each shape's q, d0 and
#     ca from its jamming runs (as in catalogue-results.tsv) and five or seven ETA around its crossing.
# Each of these 476 files records its own inputs, and this loop reruns them all (16.7 h of recorded run time,
# 4 s to 20 min each), writing *.new beside each:
#   for f in avalanche/unit-shapes/rod/*.tsv avalanche/unit-shapes/checks/*-eta*.tsv \
#            avalanche/unit-shapes/catalogue/*-eta*.tsv; do
#     set -- $(awk -F'\t' 'NR == 2 {print $5, $6, $7, $8, $13}' "$f")
#     perl avalanche/lattice/lattice_sim.pl --header --L 16 --q $1 --d0 $2 --ca $3 --eta $4 --th $5 --seed 42 \
#       > "$f.new"; done
use strict; use warnings;
use Getopt::Long;
use Time::HiRes qw(time);
$| = 1;

my %o = (L=>16, q=>6, d0=>0.283, ca=>1.2, eta=>0.02, dreg=>0.0, deliv=>'nn', labels=>'on', k=>6, th=>0.01,
         Z=>0.5, eps=>0.0, seed=>42, burn=>0, burncap=>0, M=>20000, rbudget=>0, nsnap=>10, smax=>2.0e6,
         header=>0, grid=>'', slice=>0, nslice=>1, list=>0);
GetOptions(\%o, 'L=i','q=i','d0=f','ca=f','eta=f','dreg=f','deliv=s','labels=s','k=i','th=f','Z=f','eps=f',
           'seed=i','burn=i','burncap=f','M=i','rbudget=f','nsnap=i','smax=f','header!','grid=s','slice=i',
           'nslice=i','list!','idx=s') or die "bad options\n";

my @COLS = qw(grid idx L N q d0 ca eta dreg deliv labels k th eps seed burn M t_s overflow
  A seA seA_b A_h1 A_h2 s_max sc_mom sc_mom_A2 n_p n_r nb_tot
  d_emit d_bar d_snap d_live f_dead f_loose rep_loose d_rec f_rec dw_mean f_rw
  A_B1 A_toy A_ex drift Ad2 etaAd2 D_emit
  wall wall0 G1 G2 G4 GLh xi_int mP ndom wdom fmaxdom ndom0 wdom0
  wall_f wall_l mP_f mP_l wdom_f wdom_l
  P0 P1 R0 R1 dbar0 dbar1 demit0 demit1 dold0 dold1 dw0 dw1 g_E g_R dS0 dS1
  nd_mean nd_c nd_max self_mean hist);

sub run_one {
    my %p = (%o, @_);
    my ($L,$q,$d0,$ca,$eta,$dreg,$deliv,$labels,$k,$th,$Z,$eps,$seed) =
        @p{qw(L q d0 ca eta dreg deliv labels k th Z eps seed)};
    die "nn delivery needs k = 6\n" if $deliv eq 'nn' && $k != 6;
    my $N  = $L*$L*$L;
    my $Lh = int($L/2);
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
    my @g   = (exp(-$ca/$dinit)) x $N;        # cached activation factor exp(-ca/d_i); 0 when d_i = 0
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
        $d[$i] = $dw; $g[$i] = $dw > 0 ? exp(-$ca/$dw) : 0; $rec[$i] = 1;
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
            next if rand() >= $wk * $g[$j] * $inv_theta0;          # activated trigger, as in the toy
            $npc++ if @q < $k2;                                     # a kick of the primary itself
            if ($vis[$j] != $aid) { $vis[$j] = $aid; $nd++ }
            $nself++ if $j == $i;
            my $dj = $d[$j];                                        # repack: release stored misfit ...
            my $wj = $Z * $dj * $dj / $k;
            my $bj = 6*$j;
            if ($nn) { push @q, $nb[$bj],$wj, $nb[$bj+1],$wj, $nb[$bj+2],$wj, $nb[$bj+3],$wj, $nb[$bj+4],$wj, $nb[$bj+5],$wj; }
            else     { push @q, int(rand($N)), $wj for 1..$k; }
            ($best, $bc, $tie) = (-1, 0, 0);                        # ... relabel to the neighbours' plurality ...
            for my $l (@s[@nb[$bj..$bj+5]]) {
                my $c = ++$cnt[$l];
                if ($c > $bc) { $bc = $c; $best = $l; $tie = 0 } elsif ($c == $bc) { $tie = 1 }
            }
            $cnt[$_] = 0 for @s[@nb[$bj..$bj+5]];
            $s[$j] = $best unless $tie;
            $demit += $dj; $nemit++; $nloose++ if $dj > $loose_thr;
            my $hj = $hx[$j]; $Rh[$hj]++; $demh[$hj] += $dj;
            my $dn = $dj * $shrink;                                 # ... then compact by the quantum eta
            $d[$j] = $dn; $g[$j] = exp(-$ca/$dn);
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
