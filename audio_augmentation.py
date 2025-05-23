import os
import subprocess

# Path to your base data folder
BASE_DIR = '/Users/nguyennhi/PycharmProjects/pythonProject1/test'

# Pitch shift in cents (100 cents = 1 semitone)
PITCH_SHIFT = 100  # +2 or -2 semitones
SPEED_CHANGE = 0.1  # 10% speed change

def augment_with_sox(input_path, output_path, effect_type):
    if effect_type == 'pitch_up':
        cmd = ['sox', input_path, output_path, 'pitch', str(PITCH_SHIFT)]
    elif effect_type == 'pitch_down':
        cmd = ['sox', input_path, output_path, 'pitch', str(-PITCH_SHIFT)]
    elif effect_type == 'fast':
        cmd = ['sox', input_path, output_path, 'speed', str(1 + SPEED_CHANGE)]
    elif effect_type == 'slow':
        cmd = ['sox', input_path, output_path, 'speed', str(1 - SPEED_CHANGE)]
    else:
        return  # Invalid effect
    subprocess.run(cmd)

# Iterate over each speaker folder
for person_id in os.listdir(BASE_DIR):
    person_path = os.path.join(BASE_DIR, person_id)
    if not os.path.isdir(person_path):
        continue

    # Create augmentation folders
    aug_dirs = {
        'pitch_up': os.path.join(person_path, 'pitch_up'),
        'pitch_down': os.path.join(person_path, 'pitch_down'),
        'fast': os.path.join(person_path, 'fast'),
        'slow': os.path.join(person_path, 'slow'),
    }

    for path in aug_dirs.values():
        os.makedirs(path, exist_ok=True)

    # Process each original .flac in each session subfolder
    for subfolder in os.listdir(person_path):
        subfolder_path = os.path.join(person_path, subfolder)
        if not os.path.isdir(subfolder_path) or subfolder in aug_dirs:
            continue

        for filename in os.listdir(subfolder_path):
            if not filename.endswith('.flac'):
                continue

            input_path = os.path.join(subfolder_path, filename)
            base_name = os.path.splitext(filename)[0]

            augment_with_sox(input_path, os.path.join(aug_dirs['pitch_up'], base_name + '_high.flac'), 'pitch_up')
            augment_with_sox(input_path, os.path.join(aug_dirs['pitch_down'], base_name + '_low.flac'), 'pitch_down')
            augment_with_sox(input_path, os.path.join(aug_dirs['fast'], base_name + '_fast.flac'), 'fast')
            augment_with_sox(input_path, os.path.join(aug_dirs['slow'], base_name + '_slow.flac'), 'slow')
