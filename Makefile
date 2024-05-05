include .env
export

artwork:
	find ./assets/artwork/ -type f ! -name '.gitkeep' -delete
	find ./assets/artwork/ -type d ! -path './assets/artwork/' -empty -delete

clean:
	sudo rm -rf .venv
	sudo rm -rf rpi-rgb-led-matrix
	find ./assets/artwork/ -type f ! -name '.gitkeep' -delete
	find ./assets/artwork/ -type d ! -path './assets/artwork/' -empty -delete

create:
	virtualenv -p 3.11 .venv
	$(MAKE) install-all

	sudo mkdir -p /var/cache/led-matrix-controller
	sudo chown -R root:root /var/cache/led-matrix-controller
	sudo chmod -R 777 /var/cache/led-matrix-controller

	git clone https://github.com/hzeller/rpi-rgb-led-matrix.git

	$(MAKE) -C rpi-rgb-led-matrix/bindings/python build-python PYTHON=/home/pi/led-matrix-controller/.venv/bin/python
	$(MAKE) -C rpi-rgb-led-matrix/bindings/python install-python PYTHON=/home/pi/led-matrix-controller/.venv/bin/python

	sudo rm -rf rpi-rgb-led-matrix

disable:
	sudo systemctl disable led_matrix_controller.service

enable:
	sudo systemctl enable led_matrix_controller.service

install-python:
	.venv/bin/pip install -r requirements.txt

install-service:
	sudo cp service/led_matrix_controller.service /etc/systemd/system/
	sudo systemctl daemon-reload

install-all:
	@$(MAKE) install-python
	@$(MAKE) install-service

rain:
	sudo .venv/bin/python led_matrix_controller/rain.py

restart:
	sudo systemctl restart led_matrix_controller.service

run:
	sudo .venv/bin/python led_matrix_controller/application/controller/led_matrix_controller.py

start:
	sudo systemctl start led_matrix_controller.service

stop:
	sudo systemctl stop led_matrix_controller.service

tail:
	clear && sudo journalctl -u led_matrix_controller.service -f -n 50

test:
	poetry run pytest

update:
	git add .
	git stash save "Stash before update @ $(shell date)"
	git pull --prune
	@$(MAKE) install-all


vscode-shortcut-1:
	poetry run python led_matrix_controller/rain.py
