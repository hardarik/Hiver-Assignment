import sys
import json
import argparse
from rich.console import Console
from rich.table import Table

from src.agent import AmazonSupportAgent
from src.eval_harness import run_evaluation
from src.llm_judge import validate_judge_human_agreement

console = Console()

def run_demo(query: str):
    console.print(f"\n[bold blue]Processing Query:[/bold blue] '{query}'")
    agent = AmazonSupportAgent()
    res = agent.process_message(query)

    table = Table(title="AI Support Agent Execution Result (@AmazonHelp)", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", width=22)
    table.add_column("Value / Output", style="white")

    table.add_row("Predicted Intent", f"{res['predicted_intent']} (Conf: {res['intent_confidence']:.2f})")
    table.add_row("Escalation Decision", f"[bold red]{res['escalation_decision']}[/bold red]" if res['escalation_decision'] == "ESCALATE" else f"[bold green]{res['escalation_decision']}[/bold green]")
    table.add_row("Escalation Reason", res['escalation_reason'])
    table.add_row("Draft Grounded Reply", res['draft_reply'])
    table.add_row("Top Historical RAG Precedent", res['historical_grounding'][0]['historical_customer_query'] if res['historical_grounding'] else "N/A")

    console.print(table)

def run_eval_cli():
    console.print("\n[bold green]Running Headline Benchmark Evaluation on 200 Golden Dataset Examples...[/bold green]\n")
    results = run_evaluation("data/golden_eval_set.json")

    table = Table(title="Headline Results vs Baselines (200 Golden Examples)", show_header=True, header_style="bold cyan")
    table.add_column("Model / Agent System", style="bold white")
    table.add_column("Intent Accuracy", style="yellow")
    table.add_column("Intent F1 (Macro)", style="yellow")
    table.add_column("Escalation F1", style="magenta")
    table.add_column("Reply Quality (1-5)", style="green")

    for agent_name in ["Trivial Baseline", "Simple Baseline", "Proposed System"]:
        data = results[agent_name]
        table.add_row(
            agent_name,
            f"{data['intent_accuracy']*100:.1f}%",
            f"{data['intent_f1_macro']:.4f}",
            f"{data['escalation_f1']:.4f}",
            f"{data['avg_reply_quality_score']:.2f} / 5.0"
        )

    console.print(table)

    agree = results["judge_human_agreement"]
    console.print(f"\n[bold yellow]Judge vs Human Agreement Evidence:[/bold yellow]")
    console.print(f"Sample Size: {agree['sample_size']} | % Agreement: {agree['percent_agreement']}% | Cohen's Kappa: {agree['cohens_kappa']} ({agree['interpretation']})\n")

    with open("data/results_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    console.print("[dim]Saved complete metrics report to data/results_benchmark.json[/dim]")

def main():
    parser = argparse.ArgumentParser(description="CLI Runner for @AmazonHelp AI Support Agent")
    parser.add_argument("--demo", type=str, help="Single query text to process through agent pipeline")
    parser.add_argument("--eval", action="store_true", help="Run full evaluation harness against 200 Golden Examples")
    parser.add_argument("--validate-judge", action="store_true", help="Run judge vs human agreement validation")

    args = parser.parse_args()

    if args.demo:
        run_demo(args.demo)
    elif args.eval:
        run_eval_cli()
    elif args.validate_judge:
        with open("data/golden_eval_set.json", "r", encoding="utf-8") as f:
            gset = json.load(f)
        metrics = validate_judge_human_agreement(gset)
        console.print("[bold yellow]Judge vs Human Agreement Metrics:[/bold yellow]")
        console.print(json.dumps(metrics, indent=2))
    else:
        # Default behavior: run eval
        run_eval_cli()

if __name__ == "__main__":
    main()
