#!/usr/bin/perl
# mean_field_toy.pl: the mean-field avalanche toy model of Paper II, Appendix C, Eq. (14).
#
# What it computes. N sites each hold a local free volume d_i. Kicks go to random sites (mean field).
#   - PRIMARY (a roughening event, the drive): a random site is laid loose, d_i <- d0, and emits kicks
#     of total energy z d0^2, split equally over k random receivers. One primary starts each avalanche.
#   - A KICK of energy w at site j triggers a REPACK with probability
#         p = min(1, (w/theta0) exp(-ca/d_j)),   theta0 = th z d0^2/k,
#     the Cohen-Turnbull free-volume activation, with th the trigger scale in units of the primary kick.
#     A kick that does not trigger is dropped.
#   - A REPACK at j releases the stored misfit u = d_j^2, emitting z u over k random receivers, and then
#     compacts d_j <- (1 - eta) d_j. Repacks are the route of annealing.
#   z scales the kicks and theta0 alike, so it cancels from the dynamics, as Z_1 cancels from E/theta0 in
#   Paper II. The value z = 0.5 below therefore changes no result.
# The marginal identity A d_emit^2 = 2 (Paper II, Eq. (4) with Z_1 = 1) is not built in. The column A.d2
# shows where a scan crosses it.
#
# Paper II results it supports (Sec. 5.1 and App. C):
#   law 1, self-organized marginality: the stationary state is reached from a subcritical start with no
#     tuning, and A is finite (the branching ratio 1 - 1/A stays below one);
#   law 2, the flux balance of Eq. (15): column A_flux against A;
#   law 3: eta A d_emit^2 stays nearly constant over a decade in eta, with A proportional to 1/eta
#     (at ca = 1.0 in mean-field-scan.tsv it is 0.0999, 0.099 and 0.100 at eta = 0.003, 0.007 and 0.02);
#   one-signed annealing: d_emit lies below d0 in every row, which disfavors an unannealed free volume
#     equal to d0. mean-field-scan-signed-modulus.tsv carries the scan to eta = 0.03 to 0.08, where
#     A d_emit^2 crosses 2.
#
# Output columns (one TSV line per point): ca, eta, th, d0, k, N; M avalanches measured; f_rec the fraction
# of sites laid at least once by a primary, d_rec their mean free volume (the standing mean); d_emit the
# mean free volume of the sites that repacks compact, taken before the compaction; f_loose the fraction of
# laid sites with d > 0.9 d0; A the mean avalanche size (the primary plus its repacks) and seA its standard
# error; A_E the energy-weighted amplification, the energy delivered per avalanche over z d_emit^2;
# A_flux = 1 + (d0 - d_rec)/(eta d_emit), Eq. (15) with <d_w> = d0 and the standing mean d_rec; A.d2 =
# A d_emit^2; s_max the largest avalanche; note RUNAWAY if an avalanche reached the cap of 2e6 events.
#
# Implementation: a LIFO stack of (site, kick) pairs (the totals of a branching cascade do not depend on
# the order) and a cold subcritical start d_i = min(ca/8, 0.8 d0), so the burn-in approaches the stationary
# state from below. Sites never laid by a primary still sit at the start value and are left out of d_rec.
#
# Usage (from the repository root):
#   perl avalanche/mean_field_toy.pl                          the full grid below, one TSV line per point
#   perl avalanche/mean_field_toy.pl slice I NSLICE           grid points I, I+NSLICE, ...
#   perl avalanche/mean_field_toy.pl ca eta [th d0 k N Mmeas Rbudget seed]
#        one point; defaults th 0.01, d0 0.5, k 12, N 50000, Mmeas 16000, Rbudget 2.5e6, seed 42
# The two committed scans (Perl 5.20 or later reproduces them bit for bit; 5 to 30 s per point, a few minutes
# per scan):
#   { perl avalanche/mean_field_toy.pl slice 0 3; perl avalanche/mean_field_toy.pl slice 1 3 | tail -n +2;
#     perl avalanche/mean_field_toy.pl slice 2 3 | tail -n +2;
#     perl avalanche/mean_field_toy.pl 0.45 0.001 0.01 0.16 | tail -n 1; } > avalanche/mean-field-scan.tsv
#   { perl avalanche/mean_field_toy.pl 1.2 0.03 | head -n 1;
#     for ca in 1.2 1.4 1.6 1.8 2.0; do for eta in 0.03 0.04 0.05 0.06; do
#       perl avalanche/mean_field_toy.pl $ca $eta | tail -n 1; done; done;
#     for ca in 0.3 0.4 0.5 0.6 0.8; do for eta in 0.03 0.04 0.05 0.06 0.08; do
#       perl avalanche/mean_field_toy.pl $ca $eta | tail -n 1; done; done; } > avalanche/mean-field-scan-signed-modulus.tsv
# The toy has no inputs and runs first; the lattice simulation (avalanche/lattice/) compares with it.
use strict; use warnings;
$| = 1;

my $Z = 0.5;                 # energy factor z of the kicks; it cancels from the dynamics
my $SMAX = 2.0e6;            # abort cap per avalanche (flags runaway)

sub run_one {
    my ($ca,$eta,$th,$d0,$k,$N,$Mmeas,$Rbudget,$seed) = @_;
    srand($seed);
    my $dinit = $ca/8.0; $dinit = 0.8*$d0 if $dinit > 0.8*$d0;
    my @d   = ($dinit) x $N;
    my @rec = (0) x $N;                      # recycled flag (rewritten >= once)
    my $theta0 = $th * $Z * $d0 * $d0 / $k;  # trigger scale, units of primary kick
    my $inv_theta0 = 1.0 / $theta0;
    my $shrink = 1.0 - $eta;
    my $u0 = $d0 * $d0;
    my $w0 = $Z * $u0 / $k;
    my ($demit, $nemit) = (0, 0);            # emission-weighted delta (repacks)

    # one avalanche: primary at random site, drain cascade (LIFO, flat pairs);
    # returns (s, Esum, overflow): s = rearrangements incl. primary,
    # Esum = total delivered energy
    my $avalanche = sub {
        my $i = int(rand($N));
        $d[$i] = $d0; $rec[$i] = 1;
        my @q; push @q, int(rand($N)), $w0 for 1..$k;
        my ($s, $E) = (1, $Z * $u0);
        while (@q) {
            my $wk = pop @q; my $j = pop @q;
            next if rand() >= $wk * exp(-$ca / $d[$j]) * $inv_theta0;
            # repack: release stored misfit, then compact
            my $uj = $d[$j] * $d[$j];
            my $wj = $Z * $uj / $k;
            push @q, int(rand($N)), $wj for 1..$k;
            $demit += $d[$j]; $nemit++;
            $d[$j] *= $shrink;
            $s++; $E += $Z * $uj;
            if ($s > $SMAX) { return ($s, $E, 1) }
        }
        return ($s, $E, 0);
    };

    # burn-in: load from subcritical start until stationary
    my ($btot, $bp, $over) = (0, 0, 0);
    while ($bp < 40000 && $btot < 1.5e6) {
        my ($s, $E, $ov) = $avalanche->();
        $btot += $s; $bp++; $over += $ov;
    }
    # measure
    ($demit, $nemit) = (0, 0);
    my ($M, $Ssum, $S2sum, $Esum, $Rtot, $smx) = (0,0,0,0,0,0);
    while ($M < $Mmeas && $Rtot < $Rbudget) {
        my ($s, $E, $ov) = $avalanche->();
        $over += $ov;
        $M++; $Ssum += $s; $S2sum += $s*$s; $Esum += $E; $Rtot += $s;
        $smx = $s if $s > $smx;
    }
    my $A    = $Ssum / $M;                          # event-count amplification
    my $seA  = sqrt(($S2sum/$M - $A*$A) / $M);      # SE of the mean
    my ($sr, $nr, $nl) = (0, 0, 0);                 # recycled-population stats
    for my $i (0..$N-1) {
        next unless $rec[$i];
        $nr++; $sr += $d[$i]; $nl++ if $d[$i] > 0.9*$d0;
    }
    my $drec = $nr ? $sr/$nr : 0;                   # standing delta (recycled)
    my $dem  = $nemit ? $demit/$nemit : 0;          # emission-weighted delta
    my $AE   = $dem ? ($Esum / $M) / ($Z * $dem * $dem) : 0;
    # law (ii): injection vs recycled mean, annihilation vs emission-weighted
    my $flux = $dem ? 1 + ($d0 - $drec) / ($eta * $dem) : 0;
    my $Md2  = $A * $dem * $dem;                    # marginal-identity product
    return sprintf("%.3f\t%.4g\t%.3g\t%.2f\t%d\t%.0e\t%d\t%.3f\t%.4f\t%.4f\t%.4f\t%.1f\t%.1f\t%.1f\t%.1f\t%.2f\t%d\t%s",
        $ca,$eta,$th,$d0,$k,$N,$M,($nr/$N),$drec,$dem,($nr?$nl/$nr:0),
        $A,$seA,$AE,$flux,$Md2,$smx,($over?"RUNAWAY":"ok"));
}

my $hdr = join("\t", qw(ca eta th d0 k N M f_rec d_rec d_emit f_loose A seA A_E A_flux A.d2 s_max note));

# --- grid definition ---------------------------------------------------------
my @grid;
for my $ca (0.5, 0.6, 0.8, 1.0, 1.2) {
    for my $eta (0.003, 0.007, 0.02) {
        push @grid, [$ca, $eta, 0.01, 0.5, 12, 50000];
    }
}
# robustness block around ca = 0.8, eta = 0.007
push @grid,
    [0.8, 0.007, 0.003, 0.5, 12,  50000],   # trigger scale x(1/3)
    [0.8, 0.007, 0.03,  0.5, 12,  50000],   # trigger scale x3
    [0.8, 0.007, 0.01,  0.3, 12,  50000],   # raw packing error d0 lower
    [0.8, 0.007, 0.01,  1.0, 12,  50000],   # raw packing error d0 higher
    [0.8, 0.007, 0.01,  0.5,  6,  50000],   # coordination /2
    [0.8, 0.007, 0.01,  0.5, 24,  50000],   # coordination x2
    [0.8, 0.007, 0.01,  0.5, 12, 200000];   # finite size x4

if (@ARGV && $ARGV[0] eq 'slice') {
    my (undef, $i, $ns) = @ARGV;
    print "$hdr\n";
    my $t0 = time;
    for (my $g = $i; $g < @grid; $g += $ns) {
        print run_one(@{$grid[$g]}, 16000, 2.5e6, 42), "\n";
        warn sprintf("[slice %d, %d s] done point %d\n", $i, time-$t0, $g);
    }
    exit;
}

if (@ARGV) {
    my ($ca,$eta,$th,$d0,$k,$N,$Mmeas,$Rbudget,$seed) = @ARGV;
    $th//=0.01; $d0//=0.5; $k//=12; $N//=50000;
    $Mmeas//=16000; $Rbudget//=2.5e6; $seed//=42;
    print "$hdr\n";
    print run_one($ca,$eta,$th,$d0,$k,$N,$Mmeas,$Rbudget,$seed), "\n";
    exit;
}

print "$hdr\n";
my $t0 = time;
for my $g (@grid) {
    print run_one(@$g, 16000, 2.5e6, 42), "\n";
    warn sprintf("[%d s] done ca=%.2f eta=%.3g th=%.3g d0=%.2f k=%d N=%d\n",
                 time-$t0, @$g);
}
