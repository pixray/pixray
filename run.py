import sys
import os
import pixray
import yaml

# example: python run.py cogs/hello_pixray/pixray.yaml

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_yaml file.yaml [other settings]")
        sys.exit(1)

    settings_file = sys.argv.pop(1)
    with open(settings_file, 'r') as stream:
      try:
          base_settings = yaml.safe_load(stream)
      except yaml.YAMLError as exc:
          print("YAML ERROR", exc)
          sys.exit(1)

    # This loads in default settings, but command line can override
    pixray.reset_settings()
    pixray.add_settings(**base_settings)
    settings = pixray.apply_settings()
    pixray.do_init(settings)
    pixray.do_run(settings)

if __name__ == '__main__':
    main()
