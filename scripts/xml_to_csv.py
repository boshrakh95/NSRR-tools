#!/usr/bin/env python3
"""
Convert SHHS/MrOS XML annotation files to CSV format for easier inspection.

This script helps visualize annotation timestamps to understand synchronization
issues between signal recordings and sleep stage annotations.

Usage:
    python xml_to_csv.py <xml_file_path> [--output OUTPUT_CSV] [--stages-only]
    
Examples:
    # Convert full XML to CSV
    python xml_to_csv.py /path/to/shhs1-200001-nsrr.xml
    
    # Save to specific output file
    python xml_to_csv.py /path/to/shhs1-200001-nsrr.xml --output annotations.csv
    
    # Only extract sleep stages (no respiratory events, etc.)
    python xml_to_csv.py /path/to/shhs1-200001-nsrr.xml --stages-only
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import sys
from datetime import timedelta


def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    td = timedelta(seconds=float(seconds))
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = int(td.total_seconds() % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_xml_annotations(xml_path, stages_only=False):
    """
    Parse SHHS/MrOS XML annotation file and extract events.
    
    Args:
        xml_path: Path to XML file
        stages_only: If True, only extract sleep stage events
        
    Returns:
        List of dictionaries with event information
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML file: {e}", file=sys.stderr)
        return []
    
    # Define sleep stage events
    stage_map = {
        'Stage 1 sleep|1': ('Stage1', 1),
        'Stage 2 sleep|2': ('Stage2', 2),
        'Stage 3 sleep|3': ('Stage3', 3),
        'Stage 4 sleep|4': ('Stage4', 4),
        'REM sleep|5': ('REM', 5),
        'Wake|0': ('Wake', 0),
        'Unscored|9': ('Unscored', -1)
    }
    
    events = []
    recording_start_sec = None
    recording_duration_sec = None
    
    # Parse all ScoredEvents
    for scored_event in root.findall('.//ScoredEvent'):
        event_type_elem = scored_event.find('EventType')
        event_concept_elem = scored_event.find('EventConcept')
        start_elem = scored_event.find('Start')
        duration_elem = scored_event.find('Duration')
        clock_time_elem = scored_event.find('ClockTime')
        
        if event_concept_elem is None or start_elem is None:
            continue
        
        event_concept = event_concept_elem.text.strip()
        start_sec = float(start_elem.text)
        duration_sec = float(duration_elem.text) if duration_elem is not None else 0.0
        event_type = event_type_elem.text if event_type_elem is not None else ''
        clock_time = clock_time_elem.text if clock_time_elem is not None else ''
        
        # Check if this is the recording start time marker
        if event_concept == 'Recording Start Time':
            recording_start_sec = start_sec
            recording_duration_sec = duration_sec
            if not stages_only:
                events.append({
                    'Event': 'Recording Start',
                    'Event_Type': 'Recording',
                    'Start_Seconds': start_sec,
                    'Start_Time': format_time(start_sec),
                    'Duration_Seconds': duration_sec,
                    'End_Seconds': start_sec + duration_sec,
                    'End_Time': format_time(start_sec + duration_sec),
                    'Clock_Time': clock_time,
                    'Stage_Number': '',
                    'Notes': f'Total recording duration: {duration_sec/3600:.2f} hours'
                })
            continue
        
        # Check if this is a sleep stage
        is_stage = event_concept in stage_map
        
        if stages_only and not is_stage:
            continue
        
        # Extract additional info for respiratory events
        notes = []
        if 'SpO2' in event_concept:
            nadir = scored_event.find('SpO2Nadir')
            baseline = scored_event.find('SpO2Baseline')
            if nadir is not None:
                notes.append(f"Nadir: {nadir.text}%")
            if baseline is not None:
                notes.append(f"Baseline: {baseline.text}%")
        
        # Build event dictionary
        event_dict = {
            'Event': event_concept,
            'Event_Type': event_type,
            'Start_Seconds': start_sec,
            'Start_Time': format_time(start_sec),
            'Duration_Seconds': duration_sec,
            'End_Seconds': start_sec + duration_sec,
            'End_Time': format_time(start_sec + duration_sec),
            'Clock_Time': clock_time,
            'Stage_Number': '',
            'Notes': ', '.join(notes) if notes else ''
        }
        
        # Add stage information if applicable
        if is_stage:
            stage_label, stage_num = stage_map[event_concept]
            event_dict['Event'] = stage_label
            event_dict['Stage_Number'] = stage_num
        
        events.append(event_dict)
    
    # Sort by start time
    events.sort(key=lambda x: x['Start_Seconds'])
    
    # Add sequence numbers
    stage_seq = 0
    for event in events:
        if event['Stage_Number'] != '':
            stage_seq += 1
            event['Stage_Sequence'] = stage_seq
        else:
            event['Stage_Sequence'] = ''
    
    return events, recording_start_sec, recording_duration_sec


def main():
    parser = argparse.ArgumentParser(
        description='Convert SHHS/MrOS XML annotation files to CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert full XML to CSV
  python xml_to_csv.py /scratch/nsrr_downloads/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml
  
  # Only sleep stages
  python xml_to_csv.py /path/to/file.xml --stages-only
  
  # Custom output path
  python xml_to_csv.py /path/to/file.xml --output my_annotations.csv
        """
    )
    parser.add_argument('xml_file', type=str, 
                       help='Path to XML annotation file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output CSV file path (default: same name as XML with .csv extension)')
    parser.add_argument('--stages-only', action='store_true',
                       help='Only extract sleep stage annotations (exclude respiratory events, etc.)')
    
    args = parser.parse_args()
    
    # Validate input file
    xml_path = Path(args.xml_file)
    if not xml_path.exists():
        print(f"Error: File not found: {xml_path}", file=sys.stderr)
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = xml_path.with_suffix('.csv')
    
    print(f"Converting: {xml_path.name}")
    print(f"Output to: {output_path}")
    if args.stages_only:
        print("Mode: Stages only")
    else:
        print("Mode: All events")
    print()
    
    # Parse XML
    events, recording_start, recording_duration = parse_xml_annotations(xml_path, args.stages_only)
    
    if not events:
        print("Warning: No events found in XML file", file=sys.stderr)
        return 1
    
    # Create DataFrame
    df = pd.DataFrame(events)
    
    # Reorder columns for better readability
    column_order = [
        'Event', 'Stage_Number', 'Stage_Sequence',
        'Start_Seconds', 'Start_Time', 
        'Duration_Seconds', 
        'End_Seconds', 'End_Time',
        'Event_Type', 'Clock_Time', 'Notes'
    ]
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"✓ Conversion complete!")
    print(f"  Total events: {len(events)}")
    
    if recording_start is not None:
        print(f"  Recording starts at: {recording_start}s ({format_time(recording_start)})")
    if recording_duration is not None:
        print(f"  Recording duration: {recording_duration}s ({recording_duration/3600:.2f} hours)")
    
    # Summary of sleep stages if present
    stage_events = df[df['Stage_Number'] != '']
    if not stage_events.empty:
        print(f"\n  Sleep stage events: {len(stage_events)}")
        
        # Find annotation time range
        first_stage_start = stage_events['Start_Seconds'].min()
        last_stage_end = stage_events['End_Seconds'].max()
        annotation_duration = last_stage_end - first_stage_start
        
        print(f"  First stage at: {first_stage_start}s ({format_time(first_stage_start)})")
        print(f"  Last stage ends at: {last_stage_end}s ({format_time(last_stage_end)})")
        print(f"  Annotation duration: {annotation_duration}s ({annotation_duration/3600:.2f} hours)")
        
        # Check for gap at start
        if recording_start is not None and first_stage_start > recording_start:
            gap_start = first_stage_start - recording_start
            print(f"\n  ⚠️  GAP AT START: {gap_start}s ({gap_start/60:.1f} min) between recording start and first stage")
        
        # Check for gap at end
        if recording_duration is not None:
            recording_end = recording_start + recording_duration
            if last_stage_end < recording_end:
                gap_end = recording_end - last_stage_end
                print(f"  ⚠️  GAP AT END: {gap_end}s ({gap_end/60:.1f} min) between last stage and recording end")
        
        # Stage distribution
        print(f"\n  Stage distribution:")
        stage_counts = stage_events['Event'].value_counts().sort_index()
        for stage, count in stage_counts.items():
            print(f"    {stage}: {count} epochs")
    
    print(f"\n✓ Saved to: {output_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
