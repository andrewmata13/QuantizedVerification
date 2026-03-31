To setup the environment:

conda create -n quant-env

conda activate quant-env

conda install -c conda-forge     mesa-libgl-cos7-x86_64     mesa-libegl-cos7-x86_64     libglvnd-cos7-x86_64

conda install -c conda-forge cvxpy cvxopt swiglpk

conda install -c gurobi gurobi

conda install -c mosek mosek

pip install -r requirements

pip install --upgrade onnxscript

cd nnenum_package/nnenum
pip install .