# CNVscope Visualization User Guide

## Introduction

`cnvscope_viz.py` turns one CNVscope run into an interactive HTML report: a genome overview, then one chromosome at a time with the normalized read depth, the copy-number calls drawn on top and the mapping quality along the chromosome. Here we explain how to get the two input files, build a report, open and share it, and read it.

The key points:

- A report needs two files from the same CNVscope run: the VCF and the probe dump that CNVscope writes with `--dump_probes`. sentieon-cli v2.0.0 and later writes both next to its output VCF. With the `sentieon driver`, you add `--dump_probes` to the CNVscope command (Sentieon 202503.04 or later).
- One command builds a report: `python3 scripts/cnvscope_viz.py VCF PROBES SAMPLE_report`. For a 30x genome a build takes 12 to 20 seconds and writes a folder of 55 to 80 MB.
- The report opens from disk in Chrome. It needs no web server, and no internet access unless you build it with `--cdn`. Safari opens it too, but zooming is noticeably slower there.
- The report reads CNVscope mode 0 and mode 1 output, CNVModelApply output and combined SV + CNV VCFs, and it reads the same way for all of them. By default it shows the PASS calls, which are the final call set.

The sections:

- [Install](#install): clone the repository and install the Python packages in a virtual environment.
- [Get the input files](#get-the-input-files): the VCF and the probe dump, from sentieon-cli or from two `sentieon driver` commands.
- [Build the report](#build-the-report): the build command, its log, and a folder versus a single file.
- [Open and share the report](#open-and-share-the-report): five ways to open a report, on your computer, from a server or as an attachment.
- [Reading the report](#reading-the-report): the summary tiles, the tracks and the controls.
- [Combined SV + CNV calls](#combined-sv--cnv-calls): what a combined VCF adds to the report, and its FILTER values.
- [Report builder options](#report-builder-options): every command-line option.
- [Troubleshooting](#troubleshooting): error messages and display problems, with their fixes.
- [Appendix](#appendix): the probe dump, build time and size, resolution recipes, and hooks for scripting and debugging.

## Install

| Item | What you need |
| --- | --- |
| Python | Python 3.8 or later, with numpy, plotly 5, jinja2 and vcflib (Sentieon's VCF library), all from PyPI |
| Sentieon | sentieon-cli v2.0.0 or later, or the Sentieon software, release 202503.04 or later, whose CNVscope has `--dump_probes` |
| Browser | A recent Chrome. Safari works, but zooming is noticeably slower. |

Get the scripts and install the packages once, in a virtual environment:

```bash
git clone https://github.com/Sentieon/cnv-visualization.git
cd cnv-visualization
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

We install the packages in a virtual environment because `requirements.txt` pins plotly 5. Installed into your main Python, the pin would replace a plotly 6 that other tools there may need. In a new shell, run `. .venv/bin/activate` again before you build a report, or call `.venv/bin/python3` directly.

The commands in this guide run in the `cnv-visualization` folder; from another folder, give the path to `scripts/cnvscope_viz.py`. Building and viewing a report needs no internet access, unless you build with `--cdn`.

## Get the input files

The report builder reads two files from the same CNVscope run:

- The VCF: CNVModelApply output, a combined SV + CNV VCF, or the VCF that CNVscope itself writes; bgzipped, with its index.
- The probe dump: a binary file that CNVscope writes with `--dump_probes`. It holds the normalized depth of every probe that CNVscope segments, which the report draws as its depth track. [Appendix A](#a-the-probe-dump) describes it.

### From sentieon-cli

sentieon-cli v2.0.0 and later runs CNVscope with `--dump_probes` and writes both files next to the output VCF. In the table, `OUT` is the path of the output VCF without `.vcf.gz`.

| Pipeline | VCF | Probe dump |
| --- | --- | --- |
| `dnascope-pangenome` with `--call_svs` | `OUT_cnv.vcf.gz`: CNVscope calls combined with copy-number events derived from the SV calls | `OUT_cnv.probes` |
| `dnascope-hybrid` | `OUT.cnv.vcf.gz`: CNVModelApply output | `OUT.cnv.probes` |

Both pipelines call CNVs only when the model bundle holds a CNV model (`cnv.model`). A `dnascope-hybrid` run whose output VCF is `HG002.vcf.gz`, for example, gives the report:

```bash
python3 scripts/cnvscope_viz.py HG002.cnv.vcf.gz HG002.cnv.probes HG002_report
```

[Combined SV + CNV calls](#combined-sv--cnv-calls) describes what the `dnascope-pangenome` VCF adds to the report.

### With the sentieon driver

Run CNVscope with `--dump_probes`, then score its segments with CNVModelApply. You need Sentieon 202503.04 or later and these files:

- `REF.fa`: the reference FASTA used for alignment, with its `.fai` index
- `SAMPLE.bam`: aligned, duplicate-marked reads (BAM or CRAM)
- `CNV_MODEL`: the CNV model, `cnv.model` in your Sentieon model bundle (for example `MODEL.bundle/cnv.model`); both commands take the same model
- `par_hg38.bed`: for a male human sample, the pseudo-autosomal regions (PAR) of your assembly

```bash
# 1. Segment the genome and write the probe dump
sentieon driver -t 16 -r REF.fa -i SAMPLE.bam \
    --algo CNVscope --model CNV_MODEL \
    --sex M --par par_hg38.bed \
    --dump_probes SAMPLE.probes \
    SAMPLE.raw.vcf.gz

# 2. Score the segments and assign copy numbers; keep every segment
sentieon driver -t 16 -r REF.fa \
    --algo CNVModelApply --model CNV_MODEL --emit_conf 0 \
    --vcf SAMPLE.raw.vcf.gz SAMPLE.cnv.vcf.gz

# 3. Build the report
python3 scripts/cnvscope_viz.py SAMPLE.cnv.vcf.gz SAMPLE.probes SAMPLE_report
```

When step 1 finishes, its log should include a line like `CNVscope: dumped 14507306 probes (158148 dark-masked) to SAMPLE.probes`. If it says `cannot open dump_probes file` instead, create the output directory and run step 1 again.

The CNV model sets the CNVscope mode: a mode 0 model runs mode 0 (CBS segmentation), and a mode 1 model runs mode 1 (HMM segmentation). The commands are the same for both modes, and the report's Mode tile shows which mode ran.

We run CNVModelApply with `--emit_conf 0` so that the VCF keeps every segment:

- CNVModelApply writes the PASS calls and, with `--emit_conf 0`, every segment the model rejected, with FILTER=MLrejected and the normal copy number. With its default, `--emit_conf 0.1`, it writes a rejected segment only when the model gives the normal copy number a probability of 0.9 or less.
- The report draws the rejected segments as thin grey lines. With `--emit_conf 0`, every stretch of the genome then has a segment you can hover.
- The PASS records are the same either way. The final call set is the PASS records (`bcftools view -f PASS SAMPLE.cnv.vcf.gz`).

Give CNVscope the sample's sex:

- For a male human sample, give `--sex M` and `--par` with the PAR BED of your assembly. sentieon-cli ships one per assembly in [`sentieon_cli/data`](https://github.com/Sentieon/sentieon-cli/tree/v2.0.0/sentieon_cli/data): `par_hg38.bed`, `par_hg19.bed`, `par_b37.bed` and `par_chm13.bed`. This command writes the hg38 file:

  ```bash
  printf 'chrX\t10000\t2781479\tPAR1\nchrY\t10000\t2781479\tPAR1\nchrX\t155701382\t156030895\tPAR2\nchrY\t56887902\t57217415\tPAR2\n' > par_hg38.bed
  ```

- For a female sample, give `--sex F` and leave out `--par`.
- For a non-human sample, leave out both. CNVscope then treats every chromosome as diploid.

## Build the report

```bash
python3 scripts/cnvscope_viz.py VCF PROBES OUTPUT [options]
```

- `VCF`: CNVModelApply output, a combined SV + CNV VCF, or CNVscope's own VCF, bgzipped with its `.tbi` or `.csi` index.
- `PROBES`: the probe dump from the same CNVscope run. The builder does not check that the two files match, and the sample name in the report comes from the dump.
- `OUTPUT`: a folder, or a file name ending in `.html` for one self-contained file.

A build prints a short log. This one is from the combined VCF of a 30x HG002 genome:

```
[cnvscope_viz] input: sample=HG002_30x_hs38 sex=M assembly=hg38 mode=1 · cnvscope_viz.py 1.0.0
[cnvscope_viz]        chroms=24 kept (of 195) · segments=26236 (22887 rejected, 2148 SV-derived) · output=split · advanced_tracks=off
[cnvscope_viz]        593 segments on 94 other contigs left out (--all-chroms adds them)
[cnvscope_viz] wrote HG002_report · levels=1000,10000 · tiles=430 · data=56.9 MB · total=62.1 MB · 15.2s
```

The first line names the sample, its sex, the assembly, the CNVscope mode and the version of the report builder. The second line counts the chromosomes in the report and the VCF records on them, with the rejected segments and the SV-derived calls among them. The third line counts the records on the contigs that the report leaves out, such as chrM and the unplaced contigs. The last line gives the zoom levels (bin sizes in bp), the number of data files, the size and the build time.

A report is a folder or a single file:

| | Folder (`SAMPLE_report`) | Single file (`SAMPLE_report.html`) |
| --- | --- | --- |
| Contents | `index.html`, `viewer.js`, `plotly.min.js` and a `data/` tree of small files | Everything in one file |
| Browser memory | Only the part on screen is loaded | The whole file is read when it opens |
| Best for | Whole genomes, fine bin sizes | Exomes, a few chromosomes, email attachments |

A 30x genome takes 12 to 20 seconds to build and gives a folder of 55 to 80 MB; the more records the VCF holds, the larger the folder. [Appendix B](#b-build-time-and-report-size) lists more cases.

Building again into an existing report folder replaces its data. The builder refuses to write into a non-empty folder that is not a report, unless you add `--force`; even then it writes only the report's own files (`index.html`, `viewer.js`, `plotly.min.js` and `data/`) and deletes nothing. The Sentieon logo in the page header comes from `scripts/sentieon_logo.png`, so a build makes no network request.

## Open and share the report

A report is a folder of plain files (or one `.html` file), and a browser can load it in five ways. Pick the one that matches where the report is and who needs to see it.

| Way | Use it when | You need |
| --- | --- | --- |
| 1. Open from disk | The report is on your computer | Chrome |
| 2. Local server | Opening from disk shows empty plots, or you want the address to follow your zoom | Python on the same computer |
| 3. Remote machine | The report is on a server or cluster you log into | SSH access to that machine |
| 4. Your web server | Several people should open it from one address | A web server you already run |
| 5. Single file | You want to email or attach one file | An output name ending in `.html` |

### 1. Open from disk

Open `SAMPLE_report/index.html` in Chrome (File > Open File, or drag the file into a Chrome window). Nothing else is needed: the folder holds everything the page loads. If another browser shows empty plots, use way 2.

### 2. Local server

```bash
python3 scripts/cnvscope_viz.py --serve SAMPLE_report
```

This starts a small web server for the folder and opens http://127.0.0.1:8765/index.html. It runs until you press Ctrl-C, and only your own computer can reach it. Add `--port 9000` if port 8765 is taken, or `--no-browser` to skip opening the browser.

### 3. Report on a remote machine

Copy the folder to your computer and use way 1 or 2:

```bash
scp -r user@server:/path/SAMPLE_report .
```

Or start the local server on the remote machine and forward its port to your computer:

```bash
# on the remote machine
python3 scripts/cnvscope_viz.py --serve SAMPLE_report --no-browser
# on your computer; then open http://127.0.0.1:8765/index.html
ssh -N -L 8765:127.0.0.1:8765 user@server
```

### 4. Publish on your web server

The folder needs no server-side code, so any web server that serves static files can host it: nginx, Apache, an S3 bucket or an intranet share. Copy the whole folder into the server's document root, or build straight into it, and open it by its address:

```bash
rsync -a SAMPLE_report/ webhost:/var/www/html/cnv/SAMPLE_report/
# then open https://webhost/cnv/SAMPLE_report/index.html
```

- Copy every file. A 30x genome report is about 460 files and 55 to 80 MB; missing files leave gaps in the plots.
- Publish a rebuilt report under a new folder name, so that browsers do not show cached data from the old build.
- Who can open the report is whatever the web server already allows.

### 5. Single file

Give the report builder an output name ending in `.html`:

```bash
python3 scripts/cnvscope_viz.py SAMPLE.cnv.vcf.gz SAMPLE.probes SAMPLE_report.html
```

You get one self-contained file to attach or email, and the recipient opens it in Chrome. A single file suits exomes or a few chromosomes (`--chroms chr1,chrX`). For a whole genome a folder works better: the browser reads the whole single file when it opens it, but only the part on screen from a folder.

### Good to know

- **Linking to a position.** Add `?region=chr7:55000000-56000000` to the report's address to open it at that spot. With ways 2 to 4, the address follows your zooming, so a reload keeps your place and you can copy the address to share the exact view.
- **Smaller folders.** `--cdn` leaves the plotting library (about 4.5 MB) out of the folder and loads it from cdn.plot.ly instead; everyone who opens the report then needs internet access.

## Reading the report

Each report shows a genome overview, then one chromosome at a time: normalized depth, the calls drawn on top, the mapping quality, and optional diagnostic tracks. It covers the autosomes and the sex chromosomes (`chr1` to `chr22`, `chrX` and `chrY`, or `1` to `22`, `X` and `Y`); `--all-chroms` adds chrM and the unplaced, alt and decoy contigs.

| Part | What it shows |
| --- | --- |
| Summary tiles | Sex, ploidy, mode, reference, probes (and how many are dark), segments, events, calls per Mb, median log2 ratio and MAPD. Mode is the CNVscope mode, which the builder reads from the VCF. MAPD, the median absolute difference between the log2 ratios of neighbouring probes, measures noise: lower is better, and a 30x genome with 200 bp probes gives about 0.13. Segments counts every record and splits it into PASS calls, filtered calls, rejected segments and SV-derived calls. Events and calls per Mb count the PASS calls that change copy number. All counts cover the chromosomes in the report. |
| Command line and legend | The CNVscope command from the VCF header, and a built-in guide to the report. Click to expand. |
| Genome overview | One bar per call: gains in red above the line, losses in blue below. Bar height is the distance from the normal copy number on a log2 scale, capped at ±2. Short calls get a minimum width so that they stay visible. Click a bar to open its chromosome. |
| Depth track | Each dot is the median depth ratio of the probes in one bin. Grey dots are bins of well-mapped probes, and a dot turns amber as the share of dark probes in its bin rises; a dark probe has no well-mapped reads, or a panel of normals masked it. Black lines are calls at their copy-number level, dotted teal lines are SV-derived calls, thin grey lines are segments the model rejected, and the dotted grey line is the chromosome's normal level (0.5 on a male chrX). |
| Mapping quality | Each segment's mean mapping quality (MAPQ, 0 to 60, from FORMAT/MQS) as a grey step line, for every CNVscope segment in the VCF. Low values mark regions where many reads also align elsewhere, such as segmental duplications, so the depth there is less reliable. SV-derived calls have no MAPQ. The call filters do not apply to this track. |
| Rolling SD (`--advanced`) | Noise along the chromosome: the standard deviation of the binned log2 ratio over a window of 5 bins (or the `--smooth` window, when that is 3 or more). Calls inside spikes are less reliable. |
| Dark fraction (`--advanced`) | The share of dark probes in each bin. The depth there is unreliable. |
| Segment ribbon | A bar for each call that changes copy number: blue for a loss, dark red for a homozygous deletion, red for a gain, dark rose for an amplification (CN at least normal + 2). Faded bars are calls whose FILTER is not PASS. |

**Values.** In the default linear scale, 1.0 is normal. In a diploid, CN 1 sits near 0.5, CN 3 near 1.5, CN 4 near 2.0, and a homozygous deletion at 0. In log2 scale, 0 is normal, CN 1 is −1, CN 3 is about +0.58 and CN 4 is +1. Depth is normalized across the whole sample, so on a haploid chromosome (a male chrX or chrY outside the PARs) the normal level is 0.5, and the report draws each chromosome's reference line at its own normal level. In log2 scale, bins with ratio 0 and homozygous-deletion call lines sit below the visible axis; the ribbon still marks those calls.

**Rejected segments.** The thin grey lines are the segments the CNV model rejected, when the VCF holds them; CNVModelApply writes all of them with `--emit_conf 0`. The model gives a rejected segment the normal copy number, so by default its line sits at the normal level. The *grey lines at* menu under Tracks moves the lines to each segment's measured depth. Hover the middle of a grey line for P(neutral), the model's probability that the segment has the normal copy number. Rejected segments are not calls: they have no ribbon or overview bar, the event counts leave them out, and the call filters do not apply to them.

**CNVscope's own VCF.** A report built from the VCF that CNVscope writes, before CNVModelApply, has no copy numbers. It draws each segment at its mean depth ratio and classes it by log2 ratio: a loss below −0.25, a gain above +0.2, a homozygous deletion below −1.5 and an amplification above +0.9.

| To | Do this |
| --- | --- |
| Open a chromosome | Use the Chromosome dropdown, the pills above the overview, or a bar in the overview |
| Zoom in | Drag across any track. A mostly horizontal drag zooms along the genome only; a box drag also zooms the y axis. All tracks follow, and finer bins load automatically: with the default options, views wider than 25 Mb show 10 kb bins and narrower views 1 kb bins. The chip next to the chromosome name shows the bin size and the number of points. |
| Zoom out | Double-click any track to return to the whole chromosome |
| Jump to a region | Type `chr7`, `chr7:50,000,000-60,000,000`, or `chr7:55000000` (a 1 Mb window) in Jump to region and press Enter |
| See a call's details | Hover its bar in the segment ribbon: location (POS to END, as in the VCF), CN, log2 ratio, MAPQ, QUAL, FILTER, length and ML_PROB. SV-derived calls add the SV type, CNDIFF, genotype and repeat array. A call too short to see at the current zoom responds near its midpoint. Hover a dot for its position and ratio, and an overview bar for the call's location, CN and length. |
| See a rejected segment's details | Hover the middle of its grey line: location, the model's CN and P(neutral), ML_PROB, log2 ratio, MAPQ, QUAL and length |
| See a segment's mapping quality | Hover the middle of its step on the mapping-quality track: location and MAPQ |
| Change the y axis | Pick linear or log2 under Scale. Each chromosome has a fixed y range, set from its data so that a few outliers cannot stretch it, and it does not change while you zoom. To set the range yourself, type a half-range and press Apply (1 ± value in linear, ± value in log2); switching chromosome or scale restores the automatic range. |
| Filter calls | Set Min segment size, Min QUAL, FILTER status, Event type and, for combined VCFs, Source, then press Apply. Calls that do not pass disappear from the depth track, the ribbon and the overview, and chromosomes left with no matching call turn grey in the dropdown. The filters stay in force when you switch chromosome. FILTER status starts at PASS only; choose show all to see the filtered calls, drawn faded. Gains only and losses only also hide copy-neutral segments. Reset restores the defaults. |
| Hide a track | Untick it under Tracks; the tracks below move up. The rejected segments checkbox hides the grey lines. |
| Move the grey lines | Under Tracks, set *grey lines at* to model CN (the default) or measured depth. Reset returns to model CN. |

## Combined SV + CNV calls

sentieon-cli's `dnascope-pangenome` pipeline, run with `--call_svs`, writes `OUT_cnv.vcf.gz`: the CNVscope calls combined with copy-number events derived from the SV caller's large insertions and deletions. We build its report against the probe dump of the same run:

```bash
python3 scripts/cnvscope_viz.py OUT_cnv.vcf.gz OUT_cnv.probes OUT_cnv_report
```

What the combined report adds:

- The segment ribbon gets two lanes: CNVscope calls on top, SV-derived events below.
- SV-derived events also appear as dotted teal lines on the depth track and as lighter bars in the genome overview.
- A Source filter in the sidebar shows all calls, CNVscope only, or SV-derived only.
- With FILTER status at PASS only (the default) you see the final call set. Choose show all to review the filtered records, drawn faded.
- Hover cards for SV-derived events show the SV type, CNDIFF, genotype and repeat array (REFSTART to REFSTOP). An SV-derived event's copy number is the normal copy number plus CNDIFF, the change in repeat-unit count that the SV caller reports. The dotted line marks that copy number, while the real depth change is spread over the whole repeat array.
- The Mode tile shows the mode of the CNVscope run whose calls the VCF holds.
- When the VCF holds the segments CNVModelApply rejected (FILTER=MLrejected), the report draws them in grey, as for a CNVModelApply VCF. They are not calls, and the FILTER status does not hide them.

A combined VCF marks its records with these FILTER values:

| Record | FILTER | Meaning | In the final set |
| --- | --- | --- | --- |
| CNVscope call | PASS | No SV-derived event explains it | Yes |
| CNVscope call | SVdup | The same event as an SV-derived event, which INFO `DEDUP` names | No |
| CNVscope segment | MLrejected | Rejected by the CNV model; CN is the normal copy number | No |
| CNVscope loss | NoSV | Single-end models only: a loss under 10 kb that no SV loss overlaps | No |
| SV-derived event | PASS | A converted SV call | Yes |
| SV-derived event | NoDepth | A gain of 10 kb or more where CNVscope's depth shows no gain | No |

Which of these values a combined VCF holds depends on the sentieon-cli version that wrote it. The report reads whichever values are present.

## Report builder options

`cnvscope_viz.py VCF PROBES OUTPUT [options]` builds a report; `cnvscope_viz.py --serve DIR` serves one. The defaults suit a first look at any sample.

| Option | Default | What it does |
| --- | --- | --- |
| `--advanced` | off | Adds the rolling SD and dark fraction tracks |
| `--raw-probes` | off | Adds the median probe width (about 200 bp) as the finest zoom level, for probe-level detail at close zoom; a 30x genome folder grows by about 200 MB |
| `--bin-sizes LIST` | `10000,1000` | Bin sizes in bp for the zoom levels; coarser levels are added automatically for long chromosomes |
| `--points-per-view N` | 25000 | Most points drawn at once; the view shows the finest bin size that fits the visible window |
| `--chroms LIST` | autosomes, X and Y | Only these chromosomes, for example `chr1,chrX`; the chr prefix is optional, so `1,X` works too |
| `--all-chroms` | off | Every contig: adds chrM and the unplaced, alt and decoy contigs. Not with `--chroms`. |
| `--min-chrom-probes N` | 100 | Skips contigs with fewer probes |
| `--smooth N` | 1 (off) | Rolling median over N bins (N odd) of the depth ratio |
| `--y-max H` | automatic | Fixes the log2 scale to ±H around the normal level |
| `--tile-points N` | 8192 | Bins per data file; rarely worth changing |
| `--cdn` | off | Loads the plotting library from the internet instead of bundling it |
| `--force` | off | Writes into a non-empty folder that is not a previous report; deletes nothing |
| `-v`, `--verbose` | off | Prints per-chromosome progress |
| `--serve DIR` | | Serves a report folder on 127.0.0.1 instead of building one |
| `--port N` | 8765 | Port for `--serve` |
| `--no-browser` | off | Does not open a browser with `--serve` |
| `--version` | | Prints the version of the report builder |
| `-h`, `--help` | | Lists the options |

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `cannot open dump_probes file` in the CNVscope log, or no `dumped` line | The dump's directory does not exist | Create the directory and run CNVscope again |
| `FileNotFoundError: … .vcf.gz.tbi` | The VCF has no index (usually after editing it by hand) | `tabix -p vcf SAMPLE.cnv.vcf.gz` |
| `No module named 'vcflib'` (or numpy, plotly, jinja2) | The virtual environment is not active, or a package is missing | `. .venv/bin/activate`, then `python3 -m pip install -r requirements.txt` (see [Install](#install)) |
| `not empty and not a previous report … use --force` | The report's output folder holds other files | Choose a new folder, or add `--force` |
| `no chromosome matched --chroms / --min-chrom-probes` | No listed name is in the reference (the chr prefix does not matter, but `chrM` versus `MT` does), or the listed chromosomes have too few probes | Use the reference's chromosome names |
| Empty plots when opened from disk in a browser other than Chrome | The browser blocks local scripts | Use the local server (way 2 in [Open and share the report](#open-and-share-the-report)) |
| Gaps in the plots, `tile load failed` in the browser console | The report folder was copied partially | Copy the whole folder |
| Calls do not line up with the dots anywhere | The VCF and the probe dump come from different runs | Use the VCF and the probe dump of the same CNVscope run |
| A bar in the genome overview is hard to hover | At genome scale most calls are narrower than a pixel | Click the chromosome, then hover the call's bar in the segment ribbon |

## Appendix

### A. The probe dump

CNVscope writes the probe dump when you add `--dump_probes FILE` to its command (Sentieon 202503.04 or later). The option works in mode 0 and mode 1, with or without a panel of normals, and it does not change the calls.

The dump holds every probe that survives CNVscope's preprocessing: outlier removal, GC normalization and, when CNVscope runs with a panel of normals, the panel-of-normals step. It holds what the segmenter sees. For each probe it stores:

- the start and end
- the normalized depth ratio, where 1.0 is the sample's normal copy number
- the depth ratio from well-mapped reads (MAPQ > 0)
- the fraction of well-mapped reads
- whether the probe is dark: it has no well-mapped reads, or the panel of normals masked it

The file also records the sample name, sex and ploidy, and each chromosome's ploidy blocks; on a male chrX, for example, the block outside the PARs has one copy. The docstring of `read_probes_binary()` in `scripts/cnvscope_viz.py` gives the binary layout.

The dump takes 20 bytes per probe: about 280 MB for a 30x genome with 200 bp probes (14.5 million probes), and about 7 MB for an exome (365,000 probes).

- Probes that CNVscope drops during normalization, such as chrY in a female sample, are not in the dump.
- The builder also reads dumps in format v1. Format v1 stores log ratios, so a homozygous deletion (ratio 0) is missing from such a dump.
- Keep the dump with the VCF of the same run. The builder does not check that the two match.

### B. Build time and report size

We built these reports from a 30x HG002 genome (14.5 million probes of 200 bp) and an exome (365,000 probes), with the input files in the file cache. The record counts cover the chromosomes in the report, and the sizes are in MB of 2^20 bytes, as in the build log. A busy machine or a cold file cache can make a build two to three times slower.

| Input | Records | Options | Time | Size |
| --- | --- | --- | --- | --- |
| Exome, CNVscope's own VCF | 45k | single file | 9 s | 30 MB |
| Exome, CNVscope's own VCF | 45k | single file, `--raw-probes --advanced` | 11 s | 39 MB |
| 30x genome, CNVModelApply with its default `--emit_conf` | 1.9k | folder | 13 s | 55 MB |
| 30x genome, CNVModelApply `--emit_conf 0` | 24k | folder | 15 s | 61 MB |
| 30x genome, CNVModelApply `--emit_conf 0` | 93k | folder | 20 s | 78 MB |
| 30x genome, combined SV + CNV | 26k | folder | 16 s | 62 MB |
| 30x genome, CNVModelApply `--emit_conf 0` | 93k | folder, `--raw-probes` | 31 s | 284 MB |
| 30x genome, CNVModelApply `--emit_conf 0` | 93k | folder, `--bin-sizes 200,1000,5000,25000 --advanced` | 41 s | 392 MB |
| 30x genome, chr1 and chrX only | 8.3k | single file, `--chroms chr1,chrX --raw-probes --advanced` | 12 s | 53 MB |

The builder reads the whole probe dump, about 280 MB for a 30x genome, so even a report of two chromosomes takes about 12 seconds. A build of a 30x genome peaks at about 1 GB of memory.

### C. Resolution recipes

The report stores the probe data at several bin sizes, called levels. The viewer shows the finest level that keeps the visible window within a point budget: the smallest bin size b for which the visible span divided by b is at most `--points-per-view`.

- `--bin-sizes` (default `10000,1000`) sets the bin sizes in bp, in any order. If even the coarsest level would put more than the budget on screen for the longest chromosome, the builder adds coarser levels, each 5 times the last.
- `--raw-probes` adds the median probe width as the finest level (200 bp in a standard whole-genome run), so a close zoom shows individual probes.
- `--points-per-view` (default 25000) sets the point budget. A larger budget shows finer data at wider zooms, and the browser draws more slowly.
- `--tile-points` (default 8192) sets the bins per data file.

With `--bin-sizes 200,1000,5000,25000` and the default budget, for example, windows up to 5 Mb show 200 bp bins, windows up to 25 Mb 1 kb bins, windows up to 125 Mb 5 kb bins, and the largest whole chromosomes 25 kb bins.

| Goal | Options |
| --- | --- |
| First look at a whole genome | The defaults |
| Breakpoints at probe resolution | `--raw-probes`; or `--chroms` with a single `.html` file, to keep the file small |
| Exome | A single `.html` file with `--raw-probes`, so that you can zoom into individual targets |
| Long-read runs with 10 kb windows | Start the levels at the window size, for example `--bin-sizes 10000,50000`. The builder bins probes by their start, so bins smaller than a probe only add empty bins. |

`--smooth N` takes a rolling median over N bins of the depth ratio (N odd) at every level, so its width in bp grows as you zoom out; the default of 1 means no smoothing. `--y-max H` fixes the log2 scale to ±H around the chromosome's normal level instead of the automatic range; it does not change the linear scale.

### D. Scripting and debugging hooks

- In the browser console, `cnvviz.state()` reports the chromosome, zoom level, bin size, data-file range and point count of the current view.
- `document.body.dataset.cnvReady` becomes `"1"` once the first view is drawn, which a script driving a headless browser can wait for.
- A missing data file logs `cnvscope_viz: tile load failed chr1/0/12` (chromosome / level / file number) in the browser console and leaves a gap in the plot.
- `data/manifest.json` in a report folder lists the chromosomes, their lengths and normal copy numbers, the bin sizes and the number of points in each data file.
