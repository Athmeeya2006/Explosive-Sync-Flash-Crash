.PHONY: data figures test lint cpp/build/esfc all clean

all: data figures

cpp/build/esfc:
	cmake -S cpp -B cpp/build
	cmake --build cpp/build

data: cpp/build/esfc
	python python/analysis/run_kuramoto.py --model er --n 200 --p 0.05 --k 2.0 --tmax 50 --dt 0.05 --out data/kuramoto_order.csv
	python python/analysis/early_warning.py --input data/kuramoto_order.csv --window 50 --out data/early_warning.csv
	python python/analysis/finite_size_scaling.py --n-list 50,100,200,400,800 --k-list 0.0,0.5,1.0,1.5,2.0 --n-replicas 5 --out data/finite_size.csv
	python python/analysis/flash_crash_sim.py --model er --n 200 --p 0.05 --k-high 2.5 --k-low 0.3 --t-drop 20 --out data/flash_crash.csv
	python python/analysis/hysteresis_sweep.py --out data/hysteresis.csv
	python python/analysis/lyapunov_analysis.py --out data/lyapunov.csv

figures: data
	python python/plots/plot_order_param.py --input data/kuramoto_order.csv --out figures/order_param.png
	python python/plots/plot_early_warning.py --input data/early_warning.csv --out figures/early_warning.png
	python python/plots/plot_finite_size.py --input data/finite_size.csv --out figures/finite_size.png
	python python/plots/plot_phase_diagram.py --input data/finite_size.csv --out figures/phase_diagram.png
	python python/plots/plot_flash_crash.py --input data/flash_crash.csv --out figures/flash_crash.png
	python python/plots/plot_degree_distribution.py --out figures/degree_distribution.png
	python python/plots/plot_percolation.py --out figures/percolation.png

test:
	python -m pytest python/tests -v --cov=python --cov-report=term-missing

lint:
	ruff check python/
	mypy python/ --ignore-missing-imports

clean:
	rm -rf data/*.csv data/*.npy
	rm -rf figures/*.png
	rm -rf cpp/build
