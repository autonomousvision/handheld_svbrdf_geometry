from general_settings import path_localization
import collections.abc
from collections import defaultdict
import os
import json
from utils.logging import error

def localize_settings(settings, local_paths):
    """
    Localizes the dictionary recursively, by replacing any string values containing <localization> keys.
    The replacement is done on a deep copy, which is returned.

    Inputs:
        settings        python dict
        local_paths     python dict containing the localization translations

    Outputs:
        local_settings  translated version of settings
    """
    local_settings = dict(settings)
    for path_key in local_paths:
        for data_key in local_settings:
            if isinstance(local_settings[data_key], str):
                local_settings[data_key] = local_settings[data_key].replace(path_key, local_paths[path_key])
            elif isinstance(local_settings[data_key], dict):
                local_settings[data_key] = localize_settings(local_settings[data_key], local_paths)
    return local_settings


def recursive_update(dictionary, updates):
    """
    Internal helper function to recursively update nested dictionaries in-place.
    """
    dictionary = dict(dictionary)
    for key, value in updates.items():
        if isinstance(value, collections.abc.Mapping):
            dictionary[key] = recursive_update(dictionary.get(key, {}), value)
        else:
            dictionary[key] = value
    return dictionary

class ExperimentSettings:
    def __init__(self, settings):
        self.settings = settings
    
    def localize(self):
        self.settings['local_data_settings'] = localize_settings(self.get('data_settings'), path_localization)
        self.settings['local_initialization_settings'] = localize_settings(self.get('initialization_settings'), path_localization)

    def save(self, name, index=None):
        """
        Save the given subsettings to file, depending on local_data_settings -> output_path
        """
        full_name = name + ("" if index is None else "_%d" % index)
        os.makedirs(self.get('local_data_settings')['output_path'], exist_ok=True)
        settings_file = os.path.join(self.get('local_data_settings')['output_path'], "%s.json" % full_name)
        subsettings = self.get(name, index)
        with open(settings_file, "wt") as fh:
            return json.dump(subsettings, fh, indent=4)

    def get_state_folder(self, name, index=None):
        """
        Returns the folder name for the stored state for "name".
        """
        full_name = name + ("" if index is None else "_%d" % index)
        return os.path.join(
            self.settings['local_data_settings']['output_path'],
            "stored_states",
            full_name
        )

    def get(self, name, index=None):
        """
        Get the specified subsettings.
        In the case of optimization_step settings, the default_optimization_settings are cloned, updated with the
        specified step index' settings, and returned.
        """
        subsettings = self.settings[name]
        if index is not None:
            subsettings = subsettings[index]
            if name == "optimization_steps":
                subsettings = recursive_update(self.settings["default_optimization_settings"], subsettings)
        return subsettings

    def get_shorthand(self, name, index=None):
        """
        Get a representative shorthand for the given subsettings.
        Mostly for the optimization steps subsettings, where a shorthand string is built.
        For those, capital letters represent major aspects, and small letters respective minor aspects.
        """
        if name != "optimization_steps" or index is None:
            return name
        else:
            optimization_settings = self.get(name, index)
            shorthand_base_dict = {
                "locations": "P",
                "normals": "N",
                "diffuse": "D",
                "specular": "S",
                "observation": "O",
                "light": "L",
            }
            shorthand_dict = defaultdict(lambda:[])
            for l in optimization_settings['parameters']:
                split = l.split("_")
                shorthand_dict[split[0]].append(split[1][0] if len(split) > 1 else "")
            shorthand = "".join([
                shorthand_base_dict[name]+"".join(sorted(shorthand_dict[name]))
                for name in shorthand_base_dict.keys()
                if name in shorthand_dict
            ])
            return shorthand

    def check_stored(self, name, index=None, non_critical=[]):
        """
        Checks if a stored settings file is available, reflecting that that step has been previously run.
        Returns False if no such settings are available, or True if they are available and match the provided settings.
        Raises an error if the stored settings are incompatible with the current settings.
        One can pass non-critical settings that don't necessarily need to match.
        """
        full_name = name + ("" if index is None else "_%d" % index)
        settings_file = os.path.join(self.settings['local_data_settings']['output_path'], "%s.json" % full_name)
        if os.path.exists(settings_file):
            with open(settings_file, "rt") as fh:
                stored_settings = json.load(fh)
            subsettings = self.get(name, index)
            if not (
                {k: v for k, v in subsettings.items() if k not in non_critical}
                ==
                {k: v for k, v in stored_settings.items() if k not in non_critical}
            ):
                error("The stored settings in '%s' are not compatible with the defined '%s'" % (
                    settings_file,
                    name + ("" if index is None else "[%d]" % index)
                ))
            else:
                return True
        else:
            return False
