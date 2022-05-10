
def no_recep_print_frame(oracleagent, loc):
    inst_color_count, inst_color_to_object_id = oracleagent.get_instance_seg()
    #recep_object_id = recep['object_id']

    # for each unique seg add to object dictionary if it's more visible than before
    visible_objects = []
    visible_objects_id = [] 
    visible_recep = [] 
    visible_recep_id = [] 
    
    #want it to be visible 

    c = inst_color_count
    new_inst_color_count = Counter({k: c for k, c in c.items() if c >= 10})
    
    for color, num_pixels in new_inst_color_count.most_common():
        if color in inst_color_to_object_id:
            
            
            object_id = inst_color_to_object_id[color]
            object_type = object_id.split("|")[0]
            object_metadata = oracleagent.get_obj_id_from_metadata(object_id)
            is_obj_in_recep = (object_metadata and object_metadata['parentReceptacles'] and len(object_metadata['parentReceptacles']) > 0) # and recep_object_id in object_metadata['parentReceptacles'])
            if object_id in oracleagent.receptacles.keys():
                
                visible_recep_id.append(object_id)
                visible_recep.append(oracleagent.receptacles[object_id]['num_id'])
            
            
            if object_type in oracleagent.OBJECTS and object_metadata and (not oracleagent.use_gt_relations or is_obj_in_recep):
                if object_id not in oracleagent.objects:
                    oracleagent.objects[object_id] = {
                        'object_id': object_id,
                        'object_type': object_type,
                        #'parent': recep['object_id'],
                        'loc': loc,
                        'num_pixels': num_pixels,
                        'num_id': "%s %d" % (object_type.lower() if "Sliced" not in object_id else "sliced-%s" % object_type.lower(),
                                             oracleagent.get_next_num_id(object_type, oracleagent.objects))
                    }
                elif object_id in oracleagent.objects and num_pixels > oracleagent.objects[object_id]['num_pixels']:
                    oracleagent.objects[object_id]['loc'] = loc
                    oracleagent.objects[object_id]['num_pixels'] = num_pixels

                if oracleagent.objects[object_id]['num_id'] not in oracleagent.inventory:
                    visible_objects.append(oracleagent.objects[object_id]['num_id'])
                    visible_objects_id.append(object_id)

    visible_objects_with_articles = ["a %s," % vo for vo in visible_objects]
    feedback = ""
    if len(visible_objects) > 0:
        feedback = "On the Receptacle, you see %s" % (oracleagent.fix_and_comma_in_the_end(' '.join(visible_objects_with_articles))) #recep['num_id']
    elif len(visible_objects) == 0: #not recep['closed'] and
        feedback = "On the Receptacle, you see nothing." #% (recep['num_id'])
    #pdb.set_trace()
    #print(visible_recep)
    return visible_objects, visible_objects_id, visible_recep, visible_recep_id, feedback