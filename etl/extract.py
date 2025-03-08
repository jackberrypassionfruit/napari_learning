

def get_cores_in_wip_patch(db):
    cores_in_wip_patch_query = '''\
SELECT
    part_id
FROM    [Cores2.Patch.Master]
WHERE   scrapped IS NULL
AND     start_operator IS NULL
AND     sub_process = 'Patch & Finish'
'''

    cores_in_wip_patch = db.query_to_df(
        cores_in_wip_patch_query
    )
    
    return cores_in_wip_patch