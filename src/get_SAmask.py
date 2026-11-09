import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

def get_correct_affine_from_nifti(file_path):
    img = nib.load(file_path)
    affine = img.affine.copy()
    if affine.shape != (4, 4):
        print(f"Warning: Affine shape is {affine.shape}, expected (4,4). Fixing...")
        if affine.shape == (3, 4):
            new_affine = np.eye(4)
            new_affine[:3, :] = affine
            affine = new_affine
        elif affine.shape == (4, 3):
            new_affine = np.eye(4)
            new_affine[:, :3] = affine
            affine = new_affine
    return affine, img.header


def save_decay_mask_with_correct_orientation(decay_mask, affine, header, save_path, dtype=np.float32):
    decay_mask = np.round(decay_mask, 4).astype(dtype) if dtype == np.float32 else decay_mask.astype(dtype)
    nifti_img = nib.Nifti1Image(decay_mask, affine, header)
    nib.save(nifti_img, save_path)


def get_all_ids_from_folder(data_dir, modality='DCE'):
    all_ids = []
    for label in ['Positive', 'Negative']:
        label_path = os.path.join(data_dir, label)
        if not os.path.exists(label_path):
            print(f"Warning: {label_path} does not exist")
            continue
        for id_folder in os.listdir(label_path):
            id_path = os.path.join(label_path, id_folder)
            if not os.path.isdir(id_path):
                continue
            if modality == 'DCE':
                has = glob.glob(os.path.join(id_path, f'{id_folder}_DCE_GTV.nii*'))
            elif modality == 'MLO':
                mg_path = os.path.join(id_path, 'MG')
                has = glob.glob(os.path.join(mg_path, f'*{id_folder}_*MLO_GTV.nii*'))
            else:
                raise ValueError(f"Unknown modality: {modality}")
            if has:
                all_ids.append({'ID': id_folder, 'label': label})
    print(f"Found {len(all_ids)} samples with {modality} data in {data_dir}")
    return all_ids


def create_decay_mask_adaptive(mask1, mask2, sigma_base=3, max_dist_multiplier=10):
    combined_mask = np.logical_or(mask1, mask2).astype(np.float32)

    area = np.sum(combined_mask)
    equiv_radius = np.sqrt(area / np.pi)

    if equiv_radius < 1:
        sigma = max(1.0, sigma_base * 0.5)
    elif equiv_radius < 1.5:
        sigma = sigma_base
    else:
        sigma = min(10.0, sigma_base * 1.5)

    dist_to_combined = distance_transform_edt(1 - combined_mask)
    max_dist = max_dist_multiplier * sigma
    dist_to_combined = np.clip(dist_to_combined, 0, max_dist)

    combined_decay = np.exp(- (dist_to_combined ** 2) / (2 * sigma ** 2))
    combined_decay[combined_mask == 1] = 1.0
    return combined_decay


def load_dce_masks_and_get_affine(data_dir, ID, label):
    dce_gtv_files = glob.glob(os.path.join(data_dir, label, ID, f'{ID}_DCE_GTV.nii*'))
    if not dce_gtv_files:
        raise FileNotFoundError(f"DCE GTV file not found for {ID}")
    affine, header = get_correct_affine_from_nifti(dce_gtv_files[0])

    img = nib.load(dce_gtv_files[0])
    MR_DCE_GTV = img.get_fdata()
    MR_DCE_GTV = (MR_DCE_GTV > 0.5).astype(np.float32)

    dce_gtvnd1_files = glob.glob(os.path.join(data_dir, label, ID, f'{ID}_DCE_GTVnd1.nii*'))
    if not dce_gtvnd1_files:
        raise FileNotFoundError(f"DCE GTVnd1 file not found for {ID}")
    img_nd1 = nib.load(dce_gtvnd1_files[0])
    MR_DCE_GTVnd1 = img_nd1.get_fdata()
    MR_DCE_GTVnd1 = (MR_DCE_GTVnd1 > 0.5).astype(np.float32)

    if MR_DCE_GTV.shape != MR_DCE_GTVnd1.shape:
        print(f"Warning: Shape mismatch - GTV: {MR_DCE_GTV.shape}, GTVnd1: {MR_DCE_GTVnd1.shape}")
    return MR_DCE_GTV, MR_DCE_GTVnd1, affine, header


def precompute_dce_decay_masks(data_dir, sigma_base=3, verbose=True,
                               force_recompute=False, suffix='decay'):
    all_ids = get_all_ids_from_folder(data_dir, modality='DCE')
    if not all_ids:
        print("No DCE samples found!")
        return

    if verbose:
        print(f"Total DCE samples: {len(all_ids)} | sigma_base={sigma_base}")

    failed, skipped, processed, size_stats = [], 0, 0, []

    iterator = tqdm(all_ids, desc="Computing DCE decay masks") if verbose else all_ids
    for item in iterator:
        ID, label = item['ID'], item['label']
        id_path = os.path.join(data_dir, label, ID)
        dce_mask_path = os.path.join(id_path, f'{ID}_DCE_{suffix}.nii.gz')

        if not force_recompute and os.path.exists(dce_mask_path):
            skipped += 1
            continue

        try:
            mask1, mask2, affine, header = load_dce_masks_and_get_affine(data_dir, ID, label)
            dce_decay_mask = create_decay_mask_adaptive(mask1, mask2, sigma_base=sigma_base)
            dce_decay_mask = np.clip(dce_decay_mask, 0, 1)
            save_decay_mask_with_correct_orientation(
                dce_decay_mask, affine, header, dce_mask_path, dtype=np.float16
            )
            processed += 1

            combined = np.logical_or(mask1, mask2)
            equiv_radius = np.sqrt(np.sum(combined) / np.pi)
            size_stats.append((ID, equiv_radius))
        except Exception as e:
            if verbose:
                print(f"Error processing {ID} ({label}): {e}")
            failed.append((ID, label, str(e)))

    if verbose:
        _print_summary('DCE', processed, skipped, failed, size_stats, suffix)
    return data_dir


def create_linear_decay_adaptive(mask1, mask2, distance_factor=3.0):
    combined_mask = np.logical_or(mask1, mask2).astype(np.float32)

    area = np.sum(combined_mask)
    equiv_radius = np.sqrt(area / np.pi)

    max_decay_distance = max(5, equiv_radius * distance_factor)
    max_decay_distance = min(max_decay_distance, 100)

    dist_to_combined = distance_transform_edt(1 - combined_mask)

    combined_decay = 1 - (dist_to_combined / max_decay_distance)
    combined_decay[combined_mask == 1] = 1
    combined_decay = np.clip(combined_decay, 0, 1)
    return combined_decay, max_decay_distance


def load_mlo_masks_and_get_affine(data_dir, ID, label):
    mlo_gtv_files = glob.glob(
        os.path.join(data_dir, label, ID, 'MG', f'*{ID}_*MLO_GTV.nii*')
    )
    if not mlo_gtv_files:
        raise FileNotFoundError(f"MLO GTV file not found for {ID}")
    affine, header = get_correct_affine_from_nifti(mlo_gtv_files[0])

    img = nib.load(mlo_gtv_files[0])
    MG_MLO_GTV = img.get_fdata()
    MG_MLO_GTV = (MG_MLO_GTV > 0.5).astype(np.float32)

    mlo_gtvnd1_files = glob.glob(
        os.path.join(data_dir, label, ID, 'MG', f'*{ID}_*MLO_GTVnd1.nii*')
    )
    if not mlo_gtvnd1_files:
        raise FileNotFoundError(f"MLO GTVnd1 file not found for {ID}")
    img_nd1 = nib.load(mlo_gtvnd1_files[0])
    MG_MLO_GTVnd1 = img_nd1.get_fdata()
    MG_MLO_GTVnd1 = (MG_MLO_GTVnd1 > 0.5).astype(np.float32)

    return MG_MLO_GTV, MG_MLO_GTVnd1, affine, header


def precompute_mlo_decay_masks(data_dir, distance_factor=3.0, verbose=True,
                               force_recompute=False, suffix='decay'):
    all_ids = get_all_ids_from_folder(data_dir, modality='MLO')
    if not all_ids:
        print("No MLO samples found!")
        return

    if verbose:
        print(f"Total MLO samples: {len(all_ids)} | distance_factor={distance_factor}")

    failed, skipped, processed, size_stats = [], 0, 0, []

    iterator = tqdm(all_ids, desc="Computing MLO decay masks") if verbose else all_ids
    for item in iterator:
        ID, label = item['ID'], item['label']
        id_path = os.path.join(data_dir, label, ID)
        mg_path = os.path.join(id_path, 'MG')
        mlo_mask_path = os.path.join(mg_path, f'{ID}_MLO_{suffix}.nii.gz')

        if not force_recompute and os.path.exists(mlo_mask_path):
            skipped += 1
            continue

        try:
            mask1, mask2, affine, header = load_mlo_masks_and_get_affine(data_dir, ID, label)
            mlo_decay_mask, max_dist = create_linear_decay_adaptive(
                mask1, mask2, distance_factor=distance_factor
            )
            mlo_decay_mask = np.clip(mlo_decay_mask, 0, 1)
            save_decay_mask_with_correct_orientation(
                mlo_decay_mask, affine, header, mlo_mask_path, dtype=np.float32
            )
            processed += 1

            combined = np.logical_or(mask1, mask2)
            equiv_radius = np.sqrt(np.sum(combined) / np.pi)
            size_stats.append((ID, equiv_radius, max_dist))
        except Exception as e:
            if verbose:
                print(f"Error processing {ID} ({label}): {e}")
            failed.append((ID, label, str(e)))

    if verbose:
        _print_summary('MLO', processed, skipped, failed, size_stats, suffix)
    return data_dir

def _print_summary(modality, processed, skipped, failed, size_stats, suffix):
    print(f"\n=== {modality} Done ===")
    print(f"  Processed: {processed}")
    print(f"  Skipped  : {skipped}")
    if failed:
        print(f"  Failed   : {len(failed)}")
        for ID, label, err in failed[:5]:
            print(f"    - {ID} ({label}): {err}")
    if size_stats:
        print(f"  Sample details (first 10):")
        for row in size_stats[:10]:
            if len(row) == 2:
                ID, r = row
                print(f"    {ID}: radius={r:.2f}")
            else:
                ID, r, d = row
                print(f"    {ID}: radius={r:.2f}, max_decay_dist={d:.2f}")
    print(f"  Saved as: {{ID}}/{'MG/' if modality=='MLO' else ''}{{ID}}_{modality}_{suffix}.nii.gz")

if __name__ == '__main__':
    DATASETS = {
        'train':     r'../NSLN/MR+MG20250416',
        'external':  r'../NSLN_waibu/MR+MG20250416',
        'prospect':  r'../NSLN_prospect/MR+MG20250416'
    }

    target = 'external'   # opti: 'train' / 'external' / 'prospect'

    data_dir = DATASETS[target]
    DCE_SIGMA_BASE = 3.0
    MLO_DISTANCE_FACTOR = 3.0
    SUFFIX = 'decay1'
    FORCE_RECOMPUTE = True
    RUN_DCE = True
    RUN_MLO = True

    print("=" * 60)
    print(f"dataset: {target} -> {data_dir}")
    print(f"DCE sigma_base = {DCE_SIGMA_BASE} | MLO distance_factor = {MLO_DISTANCE_FACTOR}")
    print(f"{SUFFIX} | must compute = {FORCE_RECOMPUTE}")
    print("=" * 60)

    if RUN_DCE:
        precompute_dce_decay_masks(
            data_dir=data_dir,
            sigma_base=DCE_SIGMA_BASE,
            verbose=True,
            force_recompute=FORCE_RECOMPUTE,
            suffix=SUFFIX,
        )

    if RUN_MLO:
        precompute_mlo_decay_masks(
            data_dir=data_dir,
            distance_factor=MLO_DISTANCE_FACTOR,
            verbose=True,
            force_recompute=FORCE_RECOMPUTE,
            suffix=SUFFIX,
        )

    print("\nsucess")