# MesaEvac - Mesa agent-based model of city centre evacuation

## Documentation

## Dependencies

See environment.yml

`conda env create -f environment.yml -n mesa-evac`
`conda activate mesa-evac`
`conda install --channel conda-forge --override-channels --yes fiona pyogrio`

## Python Path

From the root of the projecy
`set -a`
`source .env`

## Run

`python3 scripts/run.py --city newcastle-sm`
