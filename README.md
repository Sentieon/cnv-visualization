# CNVscope Visualization

`scripts/cnvscope_viz.py` builds an interactive HTML report from a Sentieon CNVscope run: a genome overview, then a zoomable view of each chromosome with the normalized read depth, the copy-number calls drawn on top and the mapping quality, plus optional diagnostic tracks. It reads CNVscope mode 0 and mode 1 output, CNVModelApply output, and the combined SV + CNV VCFs of sentieon-cli. A report is a folder of static files, or one `.html` file, that opens from disk in Chrome.

## Requirements

- Python 3.8 or later, with numpy, plotly 5, jinja2 and vcflib from PyPI (see Install)
- sentieon-cli v2.0.0 or later, or Sentieon 202503.04 or later, whose CNVscope writes the probe dump (`--dump_probes`)
- Chrome to view reports; Safari works, but zooming is slower

## Install

```bash
git clone https://github.com/Sentieon/cnv-visualization.git
cd cnv-visualization
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

We use a virtual environment because `requirements.txt` pins plotly 5, which would replace a plotly 6 installed in your main Python.

## Make a report

1. Get the VCF and the probe dump of one CNVscope run.
   - sentieon-cli v2.0.0 or later writes both next to its output VCF, `OUT.vcf.gz`: `OUT_cnv.vcf.gz` and `OUT_cnv.probes` from `dnascope-pangenome --call_svs`, or `OUT.cnv.vcf.gz` and `OUT.cnv.probes` from `dnascope-hybrid`.
   - With the sentieon driver, run CNVscope with `--dump_probes`, then CNVModelApply:

     ```bash
     sentieon driver -t 16 -r REF.fa -i SAMPLE.bam --algo CNVscope --model CNV_MODEL \
         --sex M --par par_hg38.bed --dump_probes SAMPLE.probes SAMPLE.raw.vcf.gz
     sentieon driver -t 16 -r REF.fa --algo CNVModelApply --model CNV_MODEL --emit_conf 0 \
         --vcf SAMPLE.raw.vcf.gz SAMPLE.cnv.vcf.gz
     ```

2. Build the report:

   ```bash
   python3 scripts/cnvscope_viz.py SAMPLE.cnv.vcf.gz SAMPLE.probes SAMPLE_report
   ```

3. Open `SAMPLE_report/index.html` in Chrome, or serve the folder with `python3 scripts/cnvscope_viz.py --serve SAMPLE_report`.

## Documentation

The [user guide](docs/user_guide.md) covers each step:

- [Install](docs/user_guide.md#install) and [Get the input files](docs/user_guide.md#get-the-input-files), including the PAR BED and the sex options
- [Build the report](docs/user_guide.md#build-the-report) and [Open and share the report](docs/user_guide.md#open-and-share-the-report)
- [Reading the report](docs/user_guide.md#reading-the-report) and [Combined SV + CNV calls](docs/user_guide.md#combined-sv--cnv-calls)
- [Report builder options](docs/user_guide.md#report-builder-options) and [Troubleshooting](docs/user_guide.md#troubleshooting)
- [Appendix](docs/user_guide.md#appendix): the probe dump, build time and size, resolution recipes, scripting hooks

## Tests

```bash
python3 -m pip install pytest
python3 -m pytest -q test/
```

Some tests need `bgzip` and `tabix` (htslib) or `node` on `PATH`; pytest skips them when the tools are missing.

## License

BSD 2-Clause License; see [LICENSE](LICENSE).

## Support

Questions and issues: support@sentieon.com
