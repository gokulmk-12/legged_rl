import numpy as np
import xml.etree.ElementTree as ET

def get_collision_info(contact, geom1, geom2):
    mask = (np.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (np.array([geom2, geom1]) == contact.geom).all(axis=1)
    if mask.any():
        idx = np.where(mask, contact.dist, 1e4).argmin()
        dist = contact.dist[idx] * mask[idx]
        normal = (dist < 0) * contact.frame[idx, 0: 3]
    else:
        dist, normal = 1000, np.array([0., 0., 0.])
    return dist, normal

def geoms_colliding(state, geom1, geom2):
    return get_collision_info(state.contact, geom1, geom2)[0] < 0

def update_hfield_size(xml_path, hfield_name, new_size):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for hfield in root.iter('hfield'):
        if hfield.get('name') == hfield_name:
            hfield.set('size', ' '.join(map(str, new_size)))
            break
    else:
        print(f"No <hfield> with name '{hfield_name}' found.")
        return

    tree.write(xml_path)
    print(f"Updated <hfield name='{hfield_name}'> size to: {new_size}")