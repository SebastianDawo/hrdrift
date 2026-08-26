# hrdrift

[![tests](https://github.com/SebastianDawo/hrdrift/actions/workflows/test.yml/badge.svg)](https://github.com/SebastianDawo/hrdrift/actions/workflows/test.yml)

Works out whether a run was below your aerobic threshold, using the heart rate
drift test from [Uphill Athlete](https://uphillathlete.com) and
[Evoke Endurance](https://evokeendurance.com). Point it at a `.tcx` or `.gpx`
and it tells you which side of 5% you landed on.

There's a [browser version](https://SebastianDawo.github.io/hrdrift/) if you'd
rather drag the file onto a page.

## The test

Warm up 15 minutes, then run 60 minutes holding **one** thing steady — either
heart rate (letting pace fall) or pace (letting heart rate rise). Flat ground
or a treadmill, chest strap. Under 5% drift means you were at or below AeT.

## Run

```bash
pip install -e .
hrdrift my-test.tcx --brief
```

```
==============================================================================
  DRIFT +5.6%/h  -  ABOVE the 5%/h threshold
  148 bpm was above your aerobic threshold. Retest around 142 bpm.
==============================================================================
```

Without `--brief` you get the full report. `--help` lists the rest.

If the run wasn't a steady effort it says so.

[@SebastianDawo](https://github.com/SebastianDawo)
