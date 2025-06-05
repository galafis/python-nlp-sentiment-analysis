"""
Main Sentiment Analysis Application
Command-line interface for sentiment analysis
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from sentiment_analyzer import SentimentAnalyzer
from preprocessing.text_cleaner import TextCleaner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Sentiment Analysis Tool')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing texts to analyze')
    parser.add_argument('--model', type=str, default='bert', 
                       choices=['bert', 'roberta', 'traditional', 'ensemble'],
                       help='Model type to use')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for processing')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize sentiment analyzer
    logger.info(f"Initializing sentiment analyzer with model: {args.model}")
    analyzer = SentimentAnalyzer(model_type=args.model)
    
    try:
        if args.text:
            # Analyze single text
            logger.info("Analyzing single text...")
            result = analyzer.analyze(args.text)
            
            print(f"\nText: {args.text}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Probabilities: {result['probabilities']}")
            
        elif args.file:
            # Analyze file
            logger.info(f"Analyzing texts from file: {args.file}")
            
            if not os.path.exists(args.file):
                logger.error(f"File not found: {args.file}")
                return
            
            # Read texts from file
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Found {len(texts)} texts to analyze")
            
            # Analyze in batches
            results = analyzer.analyze_batch(texts, batch_size=args.batch_size)
            
            # Display results
            for i, (text, result) in enumerate(zip(texts, results)):
                print(f"\n{i+1}. Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"   Sentiment: {result['sentiment']}")
                print(f"   Confidence: {result['confidence']:.3f}")
            
            # Save results if output file specified
            if args.output:
                save_results(texts, results, args.output)
                logger.info(f"Results saved to: {args.output}")
        
        else:
            # Interactive mode
            logger.info("Starting interactive mode...")
            print("Sentiment Analysis Tool - Interactive Mode")
            print("Type 'quit' to exit\n")
            
            while True:
                text = input("Enter text to analyze: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not text:
                    continue
                
                result = analyzer.analyze(text)
                print(f"Sentiment: {result['sentiment']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print()
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

def save_results(texts, results, output_file):
    """Save analysis results to file"""
    import pandas as pd
    
    # Create DataFrame with results
    data = []
    for text, result in zip(texts, results):
        data.append({
            'text': text,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'positive_prob': result['probabilities'].get('positive', 0),
            'negative_prob': result['probabilities'].get('negative', 0),
            'neutral_prob': result['probabilities'].get('neutral', 0)
        })
    
    df = pd.DataFrame(data)
    
    # Save based on file extension
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False)
    elif output_file.endswith('.json'):
        df.to_json(output_file, orient='records', indent=2)
    else:
        # Default to CSV
        df.to_csv(output_file + '.csv', index=False)

if __name__ == "__main__":
    main()

