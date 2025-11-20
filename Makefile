include .env
export

# Set AP to 0 to disable audio processor inclusion in commands
AP ?= 0

clean:
	sudo rm -rf .venv
	git submodule update --init --recursive
	cd rpi-rgb-led-matrix && git reset --hard && git clean -fd

create:
	virtualenv -p 3.11 .venv
	$(MAKE) install-all

	sudo mkdir -p /var/cache/led-matrix-controller
	sudo chown -R root:root /var/cache/led-matrix-controller
	sudo chmod -R 777 /var/cache/led-matrix-controller

	git submodule update --init --recursive

	$(MAKE) -C rpi-rgb-led-matrix/bindings/python build-python PYTHON=/home/pi/led-matrix-controller/.venv/bin/python
	$(MAKE) -C rpi-rgb-led-matrix/bindings/python install-python PYTHON=/home/pi/led-matrix-controller/.venv/bin/python

dev-update:
	@$(MAKE) update
	@$(MAKE) restart
	@$(MAKE) tail

disable:
	sudo systemctl disable led_matrix_controller.service

ifeq ($(AP), 1)
	$(MAKE) disable-ap
endif

disable-ap:
	sudo systemctl disable audio_processor.service

enable:
	sudo systemctl enable led_matrix_controller.service

ifeq ($(AP), 1)
	$(MAKE) enable-ap
endif

enable-ap:
	sudo systemctl enable audio_processor.service

install-python:
	.venv/bin/pip install -r requirements.txt

install-service:
	sudo cp service/led_matrix_controller.service /etc/systemd/system/

ifeq ($(AP), 1)
	$(MAKE) install-service-ap
else
	sudo systemctl daemon-reload
endif

install-service-ap:
	sudo cp service/audio_processor.service /etc/systemd/system/

	sudo systemctl daemon-reload

install-all:
	@$(MAKE) install-python
	@$(MAKE) install-service

rain:
	sudo .venv/bin/python led_matrix_controller/rain.py

restart:
	sudo systemctl restart led_matrix_controller.service

ifeq ($(AP), 1)
	$(MAKE) restart-ap
endif

restart-ap:
	sudo systemctl restart audio_processor.service

run:
	sudo .venv/bin/python led_matrix_controller/application/controller/led_matrix_controller.py

start:
	sudo systemctl start led_matrix_controller.service

ifeq ($(AP), 1)
	$(MAKE) start-ap
endif

start-ap:
	sudo systemctl start audio_processor.service

stop:
	sudo systemctl stop led_matrix_controller.service

ifeq ($(AP), 1)
	$(MAKE) stop-ap
endif

stop-ap:
	sudo systemctl stop audio_processor.service

tail:
	clear && sudo journalctl -u led_matrix_controller.service -f -n 100

tail-ap:
	clear && sudo journalctl -u audio_processor.service -f -n 100

test:
	poetry run pytest

update:
	git add .
	git stash save "Stash before update @ $(shell date)"
	git pull --prune
	git submodule update --init --recursive
	@$(MAKE) install-all


vscode-shortcut-1:
	poetry run python led_matrix_controller/rain.py
